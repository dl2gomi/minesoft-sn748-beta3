from typing import Sequence

from pydantic import BaseModel
from schemas.internal import Internal
from schemas.types import ImageTensor
from modules.judge.judge_pipeline import JudgePipeline, JudgeResponse


class JudgeInput(BaseModel):
    """Input for judge module."""
    pipeline: Internal[JudgePipeline]
    grid_views: Sequence[ImageTensor]
    prompt_image: ImageTensor
    seed: int


class JudgeOutput(BaseModel):
    """Output from judge module."""
    winner_index: int
    grid_view_winner: ImageTensor