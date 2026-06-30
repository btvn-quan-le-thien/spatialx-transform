from typing import Generic, TypeVar

from .transform_params import TransformationParams

TRANSFORM_T = TypeVar("TRANSFORM_T")


class ComposedParams(TransformationParams, Generic[TRANSFORM_T]):
    transforms: list[TRANSFORM_T]

    def _validate_logic(self) -> None:
        """Validate that the transforms list is not empty."""
        if not self.transforms:
            raise ValueError("transforms list cannot be empty")
