import torch
from .abstract_manifold_model_attributes import AbstractManifoldModelAttributes

__all__ = ["PolynomialAttributes"]


class PolynomialAttributes(AbstractManifoldModelAttributes):
    """
    Attributes of polynomial (quadratic) Leaspy models.

    Extends LinearAttributes by adding quadratic velocities (`v1`).

    Parameters
    ----------
    name : str
    dimension : int
    source_dimension : int
    """

    def __init__(self, name, dimension, source_dimension):
        super().__init__(name, dimension, source_dimension)

    def update(self, names_of_changed_values, values):
        """
        Update group average parameters for polynomial model.

        Adds support for 'v1' (quadratic velocities).
        """
        self._check_names(names_of_changed_values)

        compute_betas = False
        compute_positions = False
        compute_velocities_v0 = False
        compute_velocities_v1 = False

        if "all" in names_of_changed_values:
            names_of_changed_values = self.update_possibilities

        if "betas" in names_of_changed_values:
            compute_betas = True
        if "g" in names_of_changed_values:
            compute_positions = True
        if "v0" in names_of_changed_values:
            compute_velocities_v0 = True
        if "v1" in names_of_changed_values:
            compute_velocities_v1 = True

        if compute_positions:
            self._compute_positions(values)
        if compute_velocities_v0:
            self._compute_velocities(values, kind="v0")
        if compute_velocities_v1:
            self._compute_velocities(values, kind="v1")

        # Only for models with sources beyond this point
        if not self.has_sources:
            return

        if compute_betas:
            self._compute_betas(values)

        # recompute orthonormal basis if velocities changed
        if compute_velocities_v0 or compute_velocities_v1:
            self._compute_orthonormal_basis()
            self._compute_mixing_matrix()

    def _compute_positions(self, values):
        self.positions = values["g"].clone()

    def _compute_velocities(self, values, kind="v0"):
        if kind == "v0":
            self.velocities = values["v0"].clone()
        elif kind == "v1":
            self.velocities_v1 = values["v1"].clone()
        else:
            raise ValueError(f"Unknown velocity kind: {kind}")

    def _compute_orthonormal_basis(self):
        """
        Compute orthonormal basis for polynomial model.
        In linear case, it's w.r.t canonical Euclidean inner product.
        For polynomial, same approach, but consider both v0 and v1 if needed.
        """
        # You can choose to combine v0 and v1 or only use v0 as main direction
        dgamma_t0 = self.velocities  # or concatenate v0 and v1 if needed
        self._compute_Q(dgamma_t0, 1.0)
