from pydantic import AfterValidator, BeforeValidator
from typing import Annotated, Any, Literal, Tuple, TypeAlias, Union, NamedTuple
from pydantic_tensor.types import Int, Float, BFloat
from schemas.bytes import BytesAnnotation, PILImageAnnotation, TrimeshAnnotation
from schemas.tensors import TorchTensor
from PIL import Image
import trimesh

Bytes: TypeAlias = Annotated[bytes, BytesAnnotation()]
PILImage: TypeAlias = Annotated[Image.Image, PILImageAnnotation()]
TriMesh: TypeAlias = Annotated[trimesh.Trimesh, TrimeshAnnotation()]

AnyTensor: TypeAlias = TorchTensor[Any, Any]

# Integer Tensors
IntegerTensor: TypeAlias = TorchTensor[Any, Int]
IntTensor: TypeAlias = TorchTensor[Any, Literal["int32"]]
LongTensor: TypeAlias = TorchTensor[Any, Literal["int64"]]

# Floating Point Tensors
FloatingTensor: TypeAlias = TorchTensor[Any, Union[Float, BFloat]]
HalfTensor: TypeAlias = TorchTensor[Any, Literal["float16"]]
BFloatTensor: TypeAlias = TorchTensor[Any, Literal["bfloat16"]]
FloatTensor: TypeAlias = TorchTensor[Any, Literal["float32"]]
DoubleTensor: TypeAlias = TorchTensor[Any, Literal["float64"]]
BoolTensor: TypeAlias = Annotated[TorchTensor[Any, Any], BeforeValidator(lambda t: t.byte()), AfterValidator(lambda t: t.bool())]

# Image Tensors (shape-validated; assumed floating-point)
ImageChannels: TypeAlias = Literal[1, 3, 4]
ImageCHWTensor: TypeAlias = TorchTensor[Tuple[ImageChannels, int, int], Union[Float, BFloat]]
ImageHWCTensor: TypeAlias = TorchTensor[Tuple[int, int, ImageChannels], Union[Float, BFloat]]
ImagesCHWTensor: TypeAlias = TorchTensor[Tuple[int, ImageChannels, int, int], Union[Float, BFloat]]
ImagesHWCTensor: TypeAlias = TorchTensor[Tuple[int, int, int, ImageChannels], Union[Float, BFloat]]
ImageTensor: TypeAlias = ImageHWCTensor
ImagesTensor: TypeAlias = ImagesHWCTensor
AnyImageTensor: TypeAlias = Union[ImageTensor, ImageCHWTensor, ImagesTensor, ImagesCHWTensor]

class ImageSize(NamedTuple):
    height: int
    width: int
