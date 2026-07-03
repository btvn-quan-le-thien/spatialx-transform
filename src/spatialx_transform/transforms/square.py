import math

from spatialx_transform.point import Point

from .transform import Transformation, TransformationType


class Square(Transformation):
    params: None = None
    transformation_type: TransformationType = TransformationType.SQUARE

    def _point_transform(self, p: Point) -> Point:
        return Point([p.x * p.x, p.y])

    def _point_inverse(self, p: Point) -> Point:
        if p.x < 0:
            raise ValueError(
                f"Square inverse undefined for x < 0 (got x={p.x}), no real square root"
            )
        return Point([math.sqrt(p.x), p.y])
