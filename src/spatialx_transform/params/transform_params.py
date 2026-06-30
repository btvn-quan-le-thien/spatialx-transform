from abc import ABC, abstractmethod

from pydantic import BaseModel, model_validator


class TransformationParams(BaseModel, ABC):
    @abstractmethod
    def _validate_logic(self) -> None:
        """
        Subclasses must implement this to perform custom
        validation logic after field parsing.
        """
        pass

    @model_validator(mode="after")
    def run_abstract_validation(self) -> "TransformationParams":
        self._validate_logic()
        return self
