from copy import deepcopy
from types import UnionType
from typing import Annotated, Any, Optional, ClassVar, Type, TypeVar, Union, get_args, get_origin
from pydantic import BaseModel, create_model

OverridableModelT = TypeVar("OverridableModelT", bound="OverridableModel")

class OverridableModel(BaseModel):
    """
    A base model that can generate an 'Optional' counterpart for overrides
    and merge them using built-in Pydantic methods.
    """
    Overrides: ClassVar[Type[BaseModel]]

    def overrided(self, overrides_instance: Optional[BaseModel]) -> OverridableModelT:
        """
        Merges a partial Pydantic override model into the current instance's fields.
        Returns a new instance of the model with applied overrides.
        """
        if not overrides_instance:
            return self

        current_data = self.model_dump()

        for field_name in overrides_instance.model_fields_set:
            override_value = getattr(overrides_instance, field_name)
            if override_value is None:
                continue

            current_value = getattr(self, field_name, None)

            if isinstance(current_value, OverridableModel):
                if isinstance(override_value, BaseModel):
                    current_data[field_name] = current_value.overrided(override_value)
                    continue
                if isinstance(override_value, dict):
                    nested_overrides = current_value.Overrides(**override_value)
                    current_data[field_name] = current_value.overrided(nested_overrides)
                    continue

            current_data[field_name] = override_value

        return type(self)(**current_data)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs):
        super().__pydantic_init_subclass__(**kwargs)
        cls.Overrides = cls._create_optional_counterpart()

    @classmethod
    def _create_optional_counterpart(cls) -> Type[BaseModel]:
        """
        Dynamically creates the optional version of this class, preserving FieldInfo.
        """
        fields: dict[str, Any] = {}
        for name, field_info in cls.model_fields.items():
            new_field_info = deepcopy(field_info)
            new_field_info.default = None
            override_annotation = cls._to_override_annotation(field_info.annotation)
            new_field_info.annotation = Optional[override_annotation]

            fields[name] = (new_field_info.annotation, new_field_info)
        
        DynamicOverrides = create_model(f"{cls.__name__}Overrides", **fields)
        return DynamicOverrides

    @classmethod
    def _to_override_annotation(cls, annotation: Any) -> Any:
        if isinstance(annotation, type) and issubclass(annotation, OverridableModel):
            return getattr(annotation, "Overrides", annotation)

        origin = get_origin(annotation)
        if origin is None:
            return annotation

        args = get_args(annotation)
        if not args:
            return annotation

        if origin in (Union, UnionType):
            remapped_args = tuple(cls._to_override_annotation(arg) for arg in args)
            return Union[remapped_args]

        if origin is Annotated:
            remapped_annotation = cls._to_override_annotation(args[0])
            return Annotated[remapped_annotation, *args[1:]]

        remapped_args = tuple(cls._to_override_annotation(arg) for arg in args)
        try:
            if len(remapped_args) == 1:
                return origin[remapped_args[0]]
            return origin[remapped_args]
        except TypeError:
            return annotation
