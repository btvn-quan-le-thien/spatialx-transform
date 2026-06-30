from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, ClassVar, Generic, Type, TypeVar

from pydantic import BaseModel, model_validator

from spatialx_transform.params import TransformationParams
from spatialx_transform.point import Point


class TransformationType(str, Enum):
    AFFINE = "affine"
    TPS = "tps"
    COMPOSED = "composed"
    IDENTITY = "identity"

    @classmethod
    def has_value(cls, value: str) -> bool:
        return value in cls._value2member_map_


T = TypeVar("T", bound=TransformationParams)


class Transformation(BaseModel, ABC, Generic[T]):
    transformation_type: TransformationType
    params: T

    _registry: ClassVar[dict[str, Type["Transformation"]]] = {}
    IDENTITY: ClassVar["Transformation"]

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Only register concrete subclasses that define a type
        if hasattr(cls, "transformation_type") and isinstance(
            cls.transformation_type, TransformationType
        ):
            cls._registry[cls.transformation_type.value] = cls

    @model_validator(mode="wrap")
    @classmethod
    def handle_polymorphism(cls, data: Any, handler: Any) -> "Transformation":
        # If we are calling this on the base class, redirect to the subclass
        if cls is Transformation and isinstance(data, dict):
            t_type = data.get("transformation_type")

            if not isinstance(t_type, str):
                raise ValueError(f"Unknown transformation type: {t_type}")

            sub_cls = cls._registry.get(t_type)
            if sub_cls:
                return sub_cls.model_validate(data)

            raise ValueError(f"Unknown transformation type: {t_type}")

        # Otherwise, proceed with standard validation for the subclass
        return handler(data)

    @abstractmethod
    def _point_transform(self, p: Point) -> Point:
        raise NotImplementedError("Must be implemented by subclasses")

    @abstractmethod
    def _point_inverse(self, p: Point) -> Point:
        raise NotImplementedError("Must be implemented by subclasses")

    def _point_list_transform(self, ps: list[Point]) -> list[Point]:
        res = []
        for p in ps:
            res.append(self._point_transform(p))
        return res

    def _point_list_inverse(self, ps: list[Point]) -> list[Point]:
        res = []
        for p in ps:
            res.append(self._point_inverse(p))
        return res

    def transform(self, p: Point | list[Point]) -> Point | list[Point]:
        if isinstance(p, Point):
            return self._point_transform(p)
        return self._point_list_transform(p)

    def inverse(self, p: Point | list[Point]) -> Point | list[Point]:
        if isinstance(p, Point):
            return self._point_inverse(p)
        return self._point_list_inverse(p)
