from typing import Any

from pydantic import (
    BaseModel,
    field_validator,
    model_serializer,
    model_validator,
)


class Point(BaseModel):
    """A 2D point with x and y coordinates."""

    point: list[float]

    def __init__(self, point: list[float]):
        super().__init__(point=point)

    @field_validator("point", mode="before")
    @classmethod
    def _validate_point_list(cls, v: Any) -> list[float]:
        """Validate that point is a list of exactly 2 floats."""
        if isinstance(v, (tuple, list)):
            if len(v) != 2:
                raise ValueError(f"Point must have exactly 2 coordinates, got {len(v)}")
            # Convert to list of floats
            return [float(v[0]), float(v[1])]
        return v

    @model_validator(mode="before")
    @classmethod
    def _validate_from_sequence(cls, data: Any) -> Any:
        """Accept tuple/list of two numbers and convert to Point."""
        if isinstance(data, (tuple, list)):
            if len(data) != 2:
                raise ValueError(
                    f"Point must have exactly 2 coordinates, got {len(data)}"
                )
            return {"point": [float(data[0]), float(data[1])]}
        return data

    @property
    def x(self) -> float:
        return self.point[0]

    @property
    def y(self) -> float:
        return self.point[1]

    @model_serializer(mode="wrap")
    def _serialize_to_list(self, serializer: Any) -> list[float]:
        """Serialize Point to a list [x, y]."""
        return self.point


class PointList(BaseModel):
    """An array of points"""

    lst: list[float]

    @model_validator(mode="before")
    @classmethod
    def _validate_from_sequence(cls, data: Any) -> Any:
        """Accept tuple/list of even numbers and convert to PointList."""
        if isinstance(data, (tuple, list)):
            if len(data) % 2 == 1:
                raise ValueError(
                    f"Point must have exactly even lenght, got {len(data)}"
                )
            return {"lst": data}
        return data

    @model_serializer(mode="wrap")
    def _serialize_to_list(self, serializer: Any) -> list[float]:
        """Serialize Point to a list [x0, y0, x1, y1, ...]."""
        return self.lst

    @property
    def n(self) -> int:
        return len(self.lst) >> 1

    def __getitem__(self, index: int) -> Point:
        """Get a Point at the given index."""
        original_index = index
        if index < 0:
            index = self.n + index
        if index < 0 or index >= self.n:
            raise IndexError(
                f"Index {original_index} out of range for PointList with {self.n} points"
            )
        x = self.lst[index * 2]
        y = self.lst[index * 2 + 1]
        return Point([x, y])

    def __len__(self) -> int:
        """Return the number of points in the list."""
        return self.n

    def __iter__(self):
        """Iterate over points in the list."""
        for i in range(self.n):
            yield self[i]
