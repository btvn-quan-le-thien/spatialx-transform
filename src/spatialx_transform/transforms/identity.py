from spatialx_transform.point import Point

from .transform import Transformation, TransformationType


class Identity(Transformation):
    params: None = None
    transformation_type: TransformationType = TransformationType.IDENTITY

    def _point_transform(self, p: Point) -> Point:
        return p

    def _point_inverse(self, p: Point) -> Point:
        return p


Transformation.IDENTITY = Identity()
