from spatialx_transform.params import AffineParams
from spatialx_transform.point import Point

from .transform import Transformation, TransformationType


class Affine(Transformation[AffineParams]):
    params: AffineParams
    transformation_type: TransformationType = TransformationType.AFFINE

    def _point_transform(self, p: Point) -> Point:
        raise NotImplementedError()

    def _point_inverse(self, p: Point) -> Point:
        raise NotImplementedError()
