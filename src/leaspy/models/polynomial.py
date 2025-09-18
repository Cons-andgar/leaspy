import pandas as pd
import torch

from leaspy.io.data.dataset import Dataset
from leaspy.utils.functional import Exp, OrthoBasis, Sqr
from leaspy.utils.weighted_tensor import (
    unsqueeze_right,
)
from leaspy.variables.distributions import Normal
from leaspy.variables.specs import (
    Hyperparameter,
    ModelParameter,
    NamedVariables,
    PopulationLatentVariable,
    VariableNameToValueMapping,
)

from .obs_models import FullGaussianObservationModel
from .riemanian_manifold import RiemanianManifoldModel

__all__ = [
    "PolynomialInitializationMixin",
    "PolynomialModel",
]

class PolynomialInitializationMixin:
    """Compute initial values for model parameters in a quadratic polynomial model."""

    def _compute_initial_values_for_model_parameters(
        self,
        dataset: Dataset,
    ) -> VariableNameToValueMapping:
        from leaspy.models.utilities import (
            compute_linear_regression_subjects,
            get_log_velocities,
            torch_round,
        )

        df = dataset.to_pandas(apply_headers=True)
        times = df.index.get_level_values("TIME").values
        t0 = times.mean()

        # Use linear regression to estimate intercept & slope per feature
        d_regress_params = compute_linear_regression_subjects(df, max_inds=None)
        df_all = torch.tensor(df.index)  # just placeholder
        df_all_regress_params = pd.concat(d_regress_params, names=["feature"])

        # Linear term
        df_all_regress_params["position"] = (
            df_all_regress_params["intercept"] + t0 * df_all_regress_params["slope"]
        )
        df_grp = df_all_regress_params.groupby("feature", sort=False)
        positions = torch.tensor(df_grp["position"].mean().values)
        v0 = torch.tensor(df_grp["slope"].mean().values)

        # Quadratic term v1: initialize small (or 0) for all features
        v1 = torch.zeros_like(v0)

        parameters = {
            "g_mean": positions,
            "log_v0_mean": get_log_velocities(v0, self.features),
            "v1_mean": v1,
            "tau_mean": torch.tensor(t0),
            "tau_std": self.tau_std,
            "xi_std": self.xi_std,
        }

        if self.source_dimension >= 1:
            parameters["betas_mean"] = torch.zeros(
                (self.dimension - 1, self.source_dimension)
            )

        # Round to float32
        rounded_parameters = {str(p): torch_round(v.to(torch.float32)) for p, v in parameters.items()}
        return rounded_parameters


class PolynomialModel(PolynomialInitializationMixin, RiemanianManifoldModel):
    """Quadratic polynomial manifold model."""

    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)

    def get_variables_specs(self) -> NamedVariables:
        d = super().get_variables_specs()
        d.update(
            g_mean=ModelParameter.for_pop_mean("g", shape=(self.dimension,)),
            g_std=Hyperparameter(0.01),
            g=PopulationLatentVariable(Normal("g_mean", "g_std")),

            v1_mean=ModelParameter.for_pop_mean("v1", shape=(self.dimension,)),
            v1_std=Hyperparameter(0.01),
            v1=PopulationLatentVariable(Normal("v1_mean", "v1_std")),
        )
        return d

    @staticmethod
    def metric(*, g: torch.Tensor) -> torch.Tensor:
        return torch.ones_like(g)

    @classmethod
    def model_with_sources(
        cls,
        *,
        rt: torch.Tensor,
        space_shifts: torch.Tensor,
        metric,
        v0,
        v1,
        g,
    ):
        pop_s = (None, None, ...)
        rt = unsqueeze_right(rt, ndim=1)
        # Entire sum is wrapped in WeightedTensor
        return (g[pop_s] + v0[pop_s] * rt + v1[pop_s] * (rt ** 2) + space_shifts[:, None, ...]).weighted_value
