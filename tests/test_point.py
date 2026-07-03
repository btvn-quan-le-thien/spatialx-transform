"""Tests for Point and PointList."""

import pytest

from spatialx_transform.point import Point, PointList


class TestPoint:
    def test_create_from_list(self):
        p = Point([3.0, 4.0])
        assert p.x == 3.0
        assert p.y == 4.0

    def test_create_from_tuple(self):
        p = Point((1.0, 2.0))
        assert p.x == 1.0
        assert p.y == 2.0

    def test_create_from_ints(self):
        p = Point([1, 2])
        assert p.x == 1.0
        assert p.y == 2.0
        assert isinstance(p.x, float)
        assert isinstance(p.y, float)

    def test_create_invalid_length(self):
        with pytest.raises(ValueError, match="exactly 2 coordinates"):
            Point([1, 2, 3])

    def test_create_invalid_single(self):
        with pytest.raises(ValueError, match="exactly 2 coordinates"):
            Point([1])

    def test_serialize_to_list(self):
        p = Point([3.0, 4.0])
        data = p.model_dump()
        assert data == [3.0, 4.0]

    def test_round_trip(self):
        p = Point([1.5, 2.5])
        data = p.model_dump()
        p2 = Point.model_validate(data)
        assert p2.x == 1.5
        assert p2.y == 2.5

    def test_negative_coords(self):
        p = Point([-1.0, -2.0])
        assert p.x == -1.0
        assert p.y == -2.0


class TestPointList:
    def test_create_from_flat_list(self):
        pl = PointList.model_validate([1.0, 2.0, 3.0, 4.0])
        assert pl.n == 2

    def test_create_from_flat_tuple(self):
        pl = PointList.model_validate((1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        assert pl.n == 3

    def test_create_with_kwarg(self):
        pl = PointList(lst=[1.0, 2.0, 3.0, 4.0])
        assert pl.n == 2

    def test_getitem(self):
        pl = PointList.model_validate([1.0, 2.0, 3.0, 4.0])
        p = pl[0]
        assert p.x == 1.0
        assert p.y == 2.0
        p = pl[1]
        assert p.x == 3.0
        assert p.y == 4.0

    def test_negative_index(self):
        pl = PointList.model_validate([1.0, 2.0, 3.0, 4.0])
        p = pl[-1]
        assert p.x == 3.0
        assert p.y == 4.0

    def test_index_out_of_range(self):
        pl = PointList.model_validate([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(IndexError, match="out of range"):
            pl[5]

    def test_index_negative_out_of_range(self):
        pl = PointList.model_validate([1.0, 2.0, 3.0, 4.0])
        with pytest.raises(IndexError, match="out of range"):
            pl[-3]

    def test_len(self):
        pl = PointList.model_validate([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        assert len(pl) == 3

    def test_iter(self):
        pl = PointList.model_validate([1.0, 2.0, 3.0, 4.0])
        points = list(pl)
        assert len(points) == 2
        assert points[0].x == 1.0
        assert points[1].x == 3.0

    def test_odd_length_raises(self):
        with pytest.raises(ValueError, match="even lenght"):
            PointList.model_validate([1.0, 2.0, 3.0])

    def test_empty(self):
        pl = PointList.model_validate([])
        assert pl.n == 0
        assert len(pl) == 0
        assert list(pl) == []

    def test_serialize(self):
        pl = PointList.model_validate([1.0, 2.0, 3.0, 4.0])
        data = pl.model_dump()
        assert data == [1.0, 2.0, 3.0, 4.0]
