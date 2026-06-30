from spatialx_transform.point import Point

from spatialx_transform.params import TPSParams
from .transform import Transformation, TransformationType


class TPS(Transformation[TPSParams]):
    params: TPSParams
    transformation_type: TransformationType = TransformationType.TPS

    def _point_transform(self, p: Point) -> Point:
        raise NotImplementedError()

    def _point_inverse(self, p: Point) -> Point:
        raise NotImplementedError()
