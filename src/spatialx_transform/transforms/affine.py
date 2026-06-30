from spatialx_transform.params import AffineParams
from spatialx_transform.point import Point

from .transform import Transformation, TransformationType


class Affine(Transformation[AffineParams]):
    params: AffineParams
    transformation_type: TransformationType = TransformationType.AFFINE

    def _point_transform(self, p: Point) -> Point:
        A = self.params.A
        b = self.params.b
        x = A[0][0] * p.x + A[0][1] * p.y + b[0]
        y = A[1][0] * p.x + A[1][1] * p.y + b[1]
        return Point([x, y])

    def _point_inverse(self, p: Point) -> Point:
        A = self.params.A
        b = self.params.b
        det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
        if abs(det) < 1e-12:
            raise ValueError("Affine matrix is singular, cannot invert")
        inv_det = 1.0 / det
        inv_A00 = A[1][1] * inv_det
        inv_A01 = -A[0][1] * inv_det
        inv_A10 = -A[1][0] * inv_det
        inv_A11 = A[0][0] * inv_det
        dx = p.x - b[0]
        dy = p.y - b[1]
        x = inv_A00 * dx + inv_A01 * dy
        y = inv_A10 * dx + inv_A11 * dy
        return Point([x, y])
