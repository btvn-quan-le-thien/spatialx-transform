from spatialx_transform.point import PointList
from .affine_params import AffineParams
from .transform_params import TransformationParams


class TPSParams(TransformationParams):
    affine_params: AffineParams
    weights_x: list[float]
    weights_y: list[float]
    control_points: PointList

    def _validate_logic(self) -> None:
        # Validate that weights_x, weights_y, and control_points have the same length
        if len(self.weights_x) != len(self.weights_y):
            raise ValueError(
                f"weights_x and weights_y must have the same length, "
                f"got {len(self.weights_x)} and {len(self.weights_y)}"
            )
        if len(self.weights_x) != self.control_points.n:
            raise ValueError(
                f"weights_x and control_points must have the same length, "
                f"got {len(self.weights_x)} and {self.control_points.n}"
            )

    @property
    def n(self) -> int:
        """The number of control points."""
        return len(self.weights_x)
