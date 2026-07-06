"""Tests for params classes."""

import pytest

from spatialx_transform.params import (
    AffineParams,
    ComposedParams,
    TPSParams,
)
from spatialx_transform.point import PointList


class TestAffineParams:
    def test_valid(self):
        p = AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[0.0, 0.0])
        assert p.A == [[1.0, 0.0], [0.0, 1.0]]
        assert p.b == [0.0, 0.0]

    def test_identity(self):
        p = AffineParams.IDENTITY
        assert p.A == [[1.0, 0.0], [0.0, 1.0]]
        assert p.b == [0.0, 0.0]

    def test_invalid_A_rows(self):
        with pytest.raises(ValueError, match="2x2 matrix"):
            AffineParams(A=[[1.0, 0.0]], b=[0.0, 0.0])

    def test_invalid_A_cols(self):
        with pytest.raises(ValueError, match="Row 0"):
            AffineParams(A=[[1.0, 0.0, 0.0], [0.0, 1.0]], b=[0.0, 0.0])

    def test_invalid_b_length(self):
        with pytest.raises(ValueError, match="b must be a 2D vector"):
            AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[0.0])


class TestTPSParams:
    def test_valid(self):
        p = TPSParams(
            affine_params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[0.0, 0.0]),
            weights_x=[0.1, 0.2],
            weights_y=[0.3, 0.4],
            control_points=PointList.model_validate([1.0, 2.0, 3.0, 4.0]),
        )
        assert p.n == 2

    def test_mismatched_weights(self):
        with pytest.raises(ValueError, match="weights_x and weights_y"):
            TPSParams(
                affine_params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[0.0, 0.0]),
                weights_x=[0.1, 0.2],
                weights_y=[0.3],
                control_points=PointList.model_validate([1.0, 2.0, 3.0, 4.0]),
            )

    def test_mismatched_weights_and_control_points(self):
        with pytest.raises(ValueError, match="weights_x and control_points"):
            TPSParams(
                affine_params=AffineParams(A=[[1.0, 0.0], [0.0, 1.0]], b=[0.0, 0.0]),
                weights_x=[0.1],
                weights_y=[0.2],
                control_points=PointList.model_validate([1.0, 2.0, 3.0, 4.0]),
            )


class TestComposedParams:
    def test_valid(self):
        p = ComposedParams(transforms=["a", "b"])
        assert len(p.transforms) == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            ComposedParams(transforms=[])
