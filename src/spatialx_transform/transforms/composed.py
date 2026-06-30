from spatialx_transform.point import Point

from spatialx_transform.params import ComposedParams
from .transform import Transformation, TransformationType


class Composed(Transformation[ComposedParams[Transformation]]):
    params: ComposedParams[Transformation]
    transformation_type: TransformationType = TransformationType.COMPOSED

    def _point_transform(self, p: Point) -> Point:
        raise NotImplementedError()

    def _point_inverse(self, p: Point) -> Point:
        raise NotImplementedError()
