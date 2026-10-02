from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)

class PathConfig(BaseModel):
    data_root: str = "data/raw"
    cache_root: str = "data/cache"
    artifacts_root: str = "artifacts"
    dictionary_root: str = "data/dictionary"
    features_dir: str = "data/features"

class DataConfig(BaseModel):
    n_folds: int = 5
    num_labels: int = 250
    fps: int = 30

class FeatureConfig(BaseModel):
    use_z: bool = False
    use_left_hand: bool = True
    use_right_hand: bool = True
    use_pose_upper: bool = True
    use_lips: bool = True
    use_motion: bool = True
    n_frames: int = 64

    @property
    def feature_dim(self) -> int:
        from signbridge.common.landmarks import LIPS, POSE_UPPER_BODY
        dims = 3 if self.use_z else 2
        
        # Base coordinates
        n_points = 0
        if self.use_left_hand: n_points += 21
        if self.use_right_hand: n_points += 21
        if self.use_pose_upper: n_points += len(POSE_UPPER_BODY)
        if self.use_lips: n_points += len(LIPS)
        
        dim = n_points * dims
        
        # Hand shape (21 coords each)
        if self.use_left_hand: dim += 21 * dims
        if self.use_right_hand: dim += 21 * dims
        
        # Motion (first difference duplicates current feature dim)
        if self.use_motion:
            dim *= 2
            
        # Missing value indicators (1 per enabled part)
        dim += sum([self.use_left_hand, self.use_right_hand, self.use_pose_upper, self.use_lips])
        return dim

class BaseConfig(BaseModel):
    paths: PathConfig = Field(default_factory=PathConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    seed: int = 42

def load_config(path: str | Path, model_cls: type[T]) -> T:
    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return model_cls.model_validate(data)
