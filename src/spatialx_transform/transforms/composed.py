from spatialx_transform.point import Point

from spatialx_transform.params import ComposedParams
from .transform import Transformation, TransformationType


class Composed(Transformation[ComposedParams[Transformation]]):
    params: ComposedParams[Transformation]
    transformation_type: TransformationType = TransformationType.COMPOSED

    def _point_transform(self, p: Point) -> Point:
        result = p
        for t in self.params.transforms:
            result = t._point_transform(result)
        return result

    def _point_inverse(self, p: Point) -> Point:
        result = p
        for t in reversed(self.params.transforms):
            result = t._point_inverse(result)
        return result
