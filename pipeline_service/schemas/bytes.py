import base64
import io
from typing import Annotated, Any, TypeAlias

import trimesh
from PIL import Image
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler, SerializationInfo
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import core_schema


def bytes_to_base64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


def base64_to_bytes(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Expected base64 string") from exc


def image_to_bytes(value: Image.Image, image_format: str | None = None) -> bytes:
    buffer = io.BytesIO()
    fmt = image_format or value.format or "PNG"
    value.save(buffer, format=fmt)
    return buffer.getvalue()


def bytes_to_image(value: bytes) -> Image.Image:
    with io.BytesIO(value) as buffer:
        image = Image.open(buffer)
        image.load()
    return image


def image_to_base64(value: Image.Image, image_format: str | None = None) -> str:
    return bytes_to_base64(image_to_bytes(value, image_format=image_format))


def base64_to_image(value: str) -> Image.Image:
    return bytes_to_image(base64_to_bytes(value))


def _coerce_trimesh(value: Any) -> trimesh.Trimesh:
    if isinstance(value, trimesh.Trimesh):
        return value

    if isinstance(value, trimesh.Scene):
        geoms = list(value.geometry.values())
        if len(geoms) != 1:
            raise ValueError(f"Expected exactly one mesh, got {len(geoms)}")
        if not isinstance(geoms[0], trimesh.Trimesh):
            raise ValueError(f"Expected trimesh.Trimesh, got {type(geoms[0]).__name__}")
        return geoms[0]

    raise ValueError(f"Expected trimesh.Trimesh or trimesh.Scene, got {type(value).__name__}")


def trimesh_to_bytes(value: trimesh.Trimesh | trimesh.Scene, file_type: str = "glb") -> bytes:
    mesh = _coerce_trimesh(value)
    exported = mesh.export(file_type=file_type)
    if isinstance(exported, bytes):
        return exported
    if isinstance(exported, str):
        return exported.encode("utf-8")
    if isinstance(exported, bytearray):
        return bytes(exported)
    if isinstance(exported, memoryview):
        return exported.tobytes()
    if hasattr(exported, "read"):
        return exported.read()
    raise ValueError("Unsupported trimesh export type")


def bytes_to_trimesh(value: bytes, file_type: str = "glb", process: bool = False) -> trimesh.Trimesh:
    loaded = trimesh.load(file_obj=io.BytesIO(value), file_type=file_type, force=None, process=process)
    return _coerce_trimesh(loaded)


def trimesh_to_base64(value: trimesh.Trimesh | trimesh.Scene, file_type: str = "glb") -> str:
    return bytes_to_base64(trimesh_to_bytes(value, file_type=file_type))


def base64_to_trimesh(value: str, file_type: str = "glb", process: bool = False) -> trimesh.Trimesh:
    return bytes_to_trimesh(base64_to_bytes(value), file_type=file_type, process=process)


def _any_to_bytes(value: Any) -> bytes:
    if isinstance(value, bytes):
        return value
    if isinstance(value, bytearray):
        return bytes(value)
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, str):
        return base64_to_bytes(value)
    if isinstance(value, Image.Image):
        return image_to_bytes(value)
    if isinstance(value, (trimesh.Trimesh, trimesh.Scene)):
        return trimesh_to_bytes(value)
    raise ValueError(f"Unsupported bytes input type: {type(value).__name__}")


class _Base64JsonSchemaMixin:
    @staticmethod
    def __get_pydantic_json_schema__(
        schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> JsonSchemaValue:
        json_schema = handler(schema)
        json_schema["type"] = "string"
        json_schema["format"] = "base64"
        return json_schema


class BytesAnnotation(_Base64JsonSchemaMixin):
    @staticmethod
    def __get_pydantic_core_schema__(
        source: type[Any], handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        del source, handler

        def validate(value: Any) -> bytes:
            return _any_to_bytes(value)

        def serialize(value: bytes, info: SerializationInfo) -> str | bytes:
            if "json" in info.mode:
                return bytes_to_base64(value)
            return value

        validator_schema = core_schema.no_info_plain_validator_function(validate)
        return core_schema.json_or_python_schema(
            python_schema=validator_schema,
            json_schema=validator_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(serialize, info_arg=True),
        )


class PILImageAnnotation(_Base64JsonSchemaMixin):
    @staticmethod
    def __get_pydantic_core_schema__(
        source: type[Any], handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        del source, handler

        def validate(value: Any) -> Image.Image:
            if isinstance(value, Image.Image):
                return value
            return bytes_to_image(_any_to_bytes(value))

        def serialize(value: Image.Image, info: SerializationInfo) -> str | Image.Image:
            if "json" in info.mode:
                return image_to_base64(value)
            return value

        validator_schema = core_schema.no_info_plain_validator_function(validate)
        return core_schema.json_or_python_schema(
            python_schema=validator_schema,
            json_schema=validator_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(serialize, info_arg=True),
        )


class TrimeshAnnotation(_Base64JsonSchemaMixin):
    @staticmethod
    def __get_pydantic_core_schema__(
        source: type[Any], handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        del source, handler

        def validate(value: Any) -> trimesh.Trimesh:
            if isinstance(value, (trimesh.Trimesh, trimesh.Scene)):
                return _coerce_trimesh(value)
            return bytes_to_trimesh(_any_to_bytes(value))

        def serialize(value: trimesh.Trimesh, info: SerializationInfo) -> str | trimesh.Trimesh:
            if "json" in info.mode:
                return trimesh_to_base64(value)
            return value

        validator_schema = core_schema.no_info_plain_validator_function(validate)
        return core_schema.json_or_python_schema(
            python_schema=validator_schema,
            json_schema=validator_schema,
            serialization=core_schema.plain_serializer_function_ser_schema(serialize, info_arg=True),
        )


Bytes: TypeAlias = Annotated[bytes, BytesAnnotation()]
PILImage: TypeAlias = Annotated[Image.Image, PILImageAnnotation()]
Trimesh: TypeAlias = Annotated[trimesh.Trimesh, TrimeshAnnotation()]
TriMesh: TypeAlias = Trimesh
