from modules.mesh_generator.params import TrellisParams
from config.types import ModelConfig

class TrellisConfig(ModelConfig):
    model_id: str = "microsoft/TRELLIS.2-4B"
    pipeline_config_path: str = "libs/trellis2/pipeline.json"
    multiview: bool = False
    params: TrellisParams = TrellisParams()
