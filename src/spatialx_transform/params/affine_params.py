from typing import ClassVar
from .transform_params import TransformationParams


class AffineParams(TransformationParams):
    A: list[list[float]]
    b: list[float]

    IDENTITY: ClassVar["AffineParams"]

    def _validate_logic(self) -> None:
        # Validate A is a 2x2 matrix
        if len(self.A) != 2:
            raise ValueError(f"A must be a 2x2 matrix, got {len(self.A)} rows")
        for i, row in enumerate(self.A):
            if len(row) != 2:
                raise ValueError(f"Row {i} of A must have 2 elements, got {len(row)}")

        # Validate b is a 2D vector
        if len(self.b) != 2:
            raise ValueError(f"b must be a 2D vector, got {len(self.b)} elements")


AffineParams.IDENTITY = AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[0.0, 0.0])
