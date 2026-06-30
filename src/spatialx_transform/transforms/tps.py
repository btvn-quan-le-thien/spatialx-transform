import math

from spatialx_transform.params import TPSParams
from spatialx_transform.point import Point

from .transform import Transformation, TransformationType


def _tps_kernel(r: float) -> float:
    if r < 1e-10:
        return 0.0
    return r * r * math.log(r)


class TPS(Transformation[TPSParams]):
    params: TPSParams
    transformation_type: TransformationType = TransformationType.TPS

    def _point_transform(self, p: Point) -> Point:
        affine = self.params.affine_params
        wx = self.params.weights_x
        wy = self.params.weights_y
        cp = self.params.control_points
        n = cp.n

        tx = affine.A[0][0] * p.x + affine.A[0][1] * p.y + affine.b[0]
        ty = affine.A[1][0] * p.x + affine.A[1][1] * p.y + affine.b[1]

        for i in range(n):
            cx = cp.lst[i * 2]
            cy = cp.lst[i * 2 + 1]
            dx = p.x - cx
            dy = p.y - cy
            r = math.sqrt(dx * dx + dy * dy)
            U = _tps_kernel(r)
            tx += wx[i] * U
            ty += wy[i] * U

        return Point([tx, ty])

    def _point_inverse(self, p: Point) -> Point:
        result = _invert_tps_newton(
            p,
            self.params,
        )
        if result is None:
            raise ValueError("TPS inverse did not converge")
        return result


def _invert_tps_newton(
    target: Point,
    params: TPSParams,
    max_iterations: int = 20,
    tolerance: float = 1e-6,
) -> Point | None:
    affine = params.affine_params
    wx = params.weights_x
    wy = params.weights_y
    cp = params.control_points
    n = cp.n

    x = target.x
    y = target.y

    for _ in range(max_iterations):
        fx = affine.A[0][0] * x + affine.A[0][1] * y + affine.b[0]
        fy = affine.A[1][0] * x + affine.A[1][1] * y + affine.b[1]

        dxdx = affine.A[0][0]
        dxdy = affine.A[0][1]
        dydx = affine.A[1][0]
        dydy = affine.A[1][1]

        for i in range(n):
            cx = cp.lst[i * 2]
            cy = cp.lst[i * 2 + 1]
            dx = x - cx
            dy = y - cy
            r2 = dx * dx + dy * dy
            r = math.sqrt(r2)

            if r < 1e-10:
                continue

            U = _tps_kernel(r)
            fx += wx[i] * U
            fy += wy[i] * U

            dU_factor = 2.0 * math.log(r) + 1.0
            dU_dx = dU_factor * dx
            dU_dy = dU_factor * dy
            dxdx += wx[i] * dU_dx
            dxdy += wx[i] * dU_dy
            dydx += wy[i] * dU_dx
            dydy += wy[i] * dU_dy

        rx = fx - target.x
        ry = fy - target.y
        residual_norm = math.sqrt(rx * rx + ry * ry)
        if residual_norm < tolerance:
            return Point([x, y])

        det = dxdx * dydy - dxdy * dydx
        if abs(det) < 1e-10:
            return None

        delta_x = (-rx * dydy + ry * dxdy) / det
        delta_y = (rx * dydx - ry * dxdx) / det

        x += delta_x
        y += delta_y

        update_norm = math.sqrt(delta_x * delta_x + delta_y * delta_y)
        if update_norm < tolerance:
            return Point([x, y])

    return None
