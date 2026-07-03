"""Tests for transform classes."""

import math

import pytest

from spatialx_transform.params import AffineParams, TPSParams
from spatialx_transform.point import Point, PointList
from spatialx_transform.transforms import (
    Affine,
    Composed,
    Identity,
    Square,
    TPS,
    Transformation,
    TransformationType,
)


class TestTransformationType:
    def test_values(self):
        assert TransformationType.AFFINE == "affine"
        assert TransformationType.TPS == "tps"
        assert TransformationType.COMPOSED == "composed"
        assert TransformationType.IDENTITY == "identity"
        assert TransformationType.SQUARE == "square"

    def test_has_value_true(self):
        assert TransformationType.has_value("affine") is True
        assert TransformationType.has_value("tps") is True

    def test_has_value_false(self):
        assert TransformationType.has_value("unknown") is False
        assert TransformationType.has_value("") is False


class TestTransformationPolymorphism:
    def test_validate_affine(self):
        tf = Transformation.model_validate(
            {
                "transformation_type": "affine",
                "params": {"A": [[1, 0], [0, 1]], "b": [0, 0]},
            }
        )
        assert isinstance(tf, Affine)

    def test_validate_identity(self):
        tf = Transformation.model_validate(
            {"transformation_type": "identity", "params": None}
        )
        assert isinstance(tf, Identity)

    def test_validate_square(self):
        tf = Transformation.model_validate(
            {"transformation_type": "square", "params": None}
        )
        assert isinstance(tf, Square)

    def test_validate_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown transformation type"):
            Transformation.model_validate(
                {"transformation_type": "unknown", "params": None}
            )

    def test_validate_non_string_type(self):
        with pytest.raises(ValueError, match="Unknown transformation type"):
            Transformation.model_validate({"transformation_type": 123, "params": None})

    def test_validate_composed(self):
        tf = Transformation.model_validate(
            {
                "transformation_type": "composed",
                "params": {
                    "transforms": [
                        {
                            "transformation_type": "affine",
                            "params": {"A": [[1, 0], [0, 1]], "b": [0, 0]},
                        },
                        {"transformation_type": "identity", "params": None},
                    ]
                },
            }
        )
        assert isinstance(tf, Composed)
        assert len(tf.params.transforms) == 2


class TestIdentity:
    def test_transform(self):
        tf = Identity()
        p = Point([3.0, 4.0])
        result = tf.transform(p)
        assert result.x == 3.0
        assert result.y == 4.0

    def test_inverse(self):
        tf = Identity()
        p = Point([3.0, 4.0])
        result = tf.inverse(p)
        assert result.x == 3.0
        assert result.y == 4.0

    def test_transform_list(self):
        tf = Identity()
        pts = [Point([1, 2]), Point([3, 4])]
        result = tf.transform(pts)
        assert len(result) == 2
        assert result[0].x == 1.0
        assert result[1].y == 4.0

    def test_inverse_list(self):
        tf = Identity()
        pts = [Point([1, 2]), Point([3, 4])]
        result = tf.inverse(pts)
        assert len(result) == 2

    def test_round_trip(self):
        tf = Identity()
        p = Point([5.5, -3.2])
        assert tf.inverse(tf.transform(p)).x == pytest.approx(p.x)


class TestAffine:
    def test_transform_identity(self):
        tf = Affine(params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[0.0, 0.0]))
        p = Point([3.0, 4.0])
        result = tf.transform(p)
        assert result.x == 3.0
        assert result.y == 4.0

    def test_transform_translation(self):
        tf = Affine(params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[10.0, 20.0]))
        p = Point([3.0, 4.0])
        result = tf.transform(p)
        assert result.x == 13.0
        assert result.y == 24.0

    def test_transform_rotation(self):
        angle = math.pi / 4
        c, s = math.cos(angle), math.sin(angle)
        tf = Affine(params=AffineParams(A=[[c, -s], [s, c]], b=[0.0, 0.0]))
        p = Point([1.0, 0.0])
        result = tf.transform(p)
        assert result.x == pytest.approx(c, abs=1e-10)
        assert result.y == pytest.approx(s, abs=1e-10)

    def test_transform_scale(self):
        tf = Affine(params=AffineParams(A=[[2.0, 0.0], [0.0, 3.0]], b=[0.0, 0.0]))
        p = Point([1.0, 1.0])
        result = tf.transform(p)
        assert result.x == 2.0
        assert result.y == 3.0

    def test_inverse_identity(self):
        tf = Affine(params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[0.0, 0.0]))
        p = Point([3.0, 4.0])
        result = tf.inverse(p)
        assert result.x == 3.0
        assert result.y == 4.0

    def test_inverse_translation(self):
        tf = Affine(params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[10.0, 20.0]))
        p = Point([13.0, 24.0])
        result = tf.inverse(p)
        assert result.x == 3.0
        assert result.y == 4.0

    def test_inverse_rotation(self):
        angle = math.pi / 4
        c, s = math.cos(angle), math.sin(angle)
        tf = Affine(params=AffineParams(A=[[c, -s], [s, c]], b=[0.0, 0.0]))
        p = Point([c, s])
        result = tf.inverse(p)
        assert result.x == pytest.approx(1.0, abs=1e-10)
        assert result.y == pytest.approx(0.0, abs=1e-10)

    def test_round_trip(self):
        tf = Affine(params=AffineParams(A=[[2.0, 1.0], [0.5, 3.0]], b=[5.0, -2.0]))
        p = Point([3.0, 7.0])
        fwd = tf.transform(p)
        inv = tf.inverse(fwd)
        assert inv.x == pytest.approx(p.x, abs=1e-6)
        assert inv.y == pytest.approx(p.y, abs=1e-6)

    def test_singular_matrix_raises(self):
        tf = Affine(params=AffineParams(A=[[1.0, 2.0], [2.0, 4.0]], b=[0.0, 0.0]))
        with pytest.raises(ValueError, match="singular"):
            tf.inverse(Point([1.0, 2.0]))

    def test_transform_list(self):
        tf = Affine(params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[1.0, 1.0]))
        pts = [Point([0, 0]), Point([1, 1])]
        result = tf.transform(pts)
        assert result[0].x == 1.0
        assert result[1].y == 2.0


class TestSquare:
    def test_transform(self):
        tf = Square()
        p = Point([3.0, 4.0])
        result = tf.transform(p)
        assert result.x == 9.0
        assert result.y == 4.0

    def test_transform_zero(self):
        tf = Square()
        p = Point([0.0, 5.0])
        result = tf.transform(p)
        assert result.x == 0.0
        assert result.y == 5.0

    def test_transform_negative(self):
        tf = Square()
        p = Point([-2.0, 3.0])
        result = tf.transform(p)
        assert result.x == 4.0
        assert result.y == 3.0

    def test_inverse(self):
        tf = Square()
        p = Point([9.0, 4.0])
        result = tf.inverse(p)
        assert result.x == 3.0
        assert result.y == 4.0

    def test_inverse_zero(self):
        tf = Square()
        result = tf.inverse(Point([0.0, 5.0]))
        assert result.x == 0.0
        assert result.y == 5.0

    def test_inverse_negative_raises(self):
        tf = Square()
        with pytest.raises(ValueError, match="x < 0"):
            tf.inverse(Point([-1.0, 0.0]))

    def test_round_trip(self):
        tf = Square()
        p = Point([5.0, 3.0])
        fwd = tf.transform(p)
        inv = tf.inverse(fwd)
        assert inv.x == pytest.approx(p.x, abs=1e-6)
        assert inv.y == pytest.approx(p.y, abs=1e-6)


class TestTPS:
    def _make_simple_tps(self):
        return TPS(
            params=TPSParams(
                affine_params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[0.0, 0.0]),
                weights_x=[0.0, 0.0],
                weights_y=[0.0, 0.0],
                control_points=PointList.model_validate([0.0, 0.0, 10.0, 10.0]),
            )
        )

    def test_transform_zero_weights(self):
        tf = self._make_simple_tps()
        p = Point([3.0, 4.0])
        result = tf.transform(p)
        assert result.x == 3.0
        assert result.y == 4.0

    def test_inverse_zero_weights(self):
        tf = self._make_simple_tps()
        p = Point([3.0, 4.0])
        result = tf.inverse(p)
        assert result.x == pytest.approx(3.0, abs=1e-6)
        assert result.y == pytest.approx(4.0, abs=1e-6)

    def test_round_trip(self):
        tf = self._make_simple_tps()
        p = Point([5.0, 7.0])
        fwd = tf.transform(p)
        inv = tf.inverse(fwd)
        assert inv.x == pytest.approx(p.x, abs=1e-4)
        assert inv.y == pytest.approx(p.y, abs=1e-4)

    def test_transform_at_control_point(self):
        tf = self._make_simple_tps()
        p = Point([0.0, 0.0])
        result = tf.transform(p)
        assert result.x == 0.0
        assert result.y == 0.0

    def test_transform_nonzero_weights(self):
        tf = TPS(
            params=TPSParams(
                affine_params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[0.0, 0.0]),
                weights_x=[0.1],
                weights_y=[0.2],
                control_points=PointList.model_validate([5.0, 5.0]),
            )
        )
        p = Point([5.0, 5.0])
        result = tf.transform(p)
        assert result.x == 5.0
        assert result.y == 5.0

    def test_kernel_zero(self):
        from spatialx_transform.transforms.tps import _tps_kernel

        assert _tps_kernel(0.0) == 0.0
        assert _tps_kernel(1e-11) == 0.0

    def test_kernel_nonzero(self):
        from spatialx_transform.transforms.tps import _tps_kernel

        r = 2.0
        expected = r * r * math.log(r)
        assert _tps_kernel(r) == pytest.approx(expected)


class TestComposed:
    def test_transform_two_affines(self):
        tf = Composed(
            params={
                "transforms": [
                    Affine(
                        params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[1.0, 0.0])
                    ),
                    Affine(
                        params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[0.0, 2.0])
                    ),
                ]
            }
        )
        p = Point([0.0, 0.0])
        result = tf.transform(p)
        assert result.x == 1.0
        assert result.y == 2.0

    def test_inverse_reverses_order(self):
        tf = Composed(
            params={
                "transforms": [
                    Affine(
                        params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[1.0, 0.0])
                    ),
                    Affine(
                        params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[0.0, 2.0])
                    ),
                ]
            }
        )
        p = Point([1.0, 2.0])
        result = tf.inverse(p)
        assert result.x == pytest.approx(0.0, abs=1e-6)
        assert result.y == pytest.approx(0.0, abs=1e-6)

    def test_round_trip(self):
        tf = Composed(
            params={
                "transforms": [
                    Affine(
                        params=AffineParams(A=[[2.0, 0.0], [0.0, 3.0]], b=[1.0, 1.0])
                    ),
                    Square(),
                ]
            }
        )
        p = Point([4.0, 2.0])
        fwd = tf.transform(p)
        inv = tf.inverse(fwd)
        assert inv.x == pytest.approx(p.x, abs=1e-4)
        assert inv.y == pytest.approx(p.y, abs=1e-4)

    def test_composed_with_identity(self):
        tf = Composed(
            params={
                "transforms": [
                    Identity(),
                    Affine(
                        params=AffineParams(A=[[2.0, 0.0], [0.0, 2.0]], b=[0.0, 0.0])
                    ),
                ]
            }
        )
        p = Point([3.0, 4.0])
        result = tf.transform(p)
        assert result.x == 6.0
        assert result.y == 8.0

    def test_transform_list(self):
        tf = Composed(
            params={
                "transforms": [
                    Affine(
                        params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[1.0, 1.0])
                    ),
                ]
            }
        )
        pts = [Point([0, 0]), Point([1, 1])]
        result = tf.transform(pts)
        assert result[0].x == 1.0
        assert result[1].y == 2.0
