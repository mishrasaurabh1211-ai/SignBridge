import os
from pathlib import Path

# Create directories
dirs = [
    "signbridge/common", "signbridge/data", "signbridge/features", 
    "signbridge/models", "signbridge/recognition", "signbridge/generation", 
    "signbridge/api", "tests", "configs", "docs"
]
for d in dirs:
    os.makedirs(d, exist_ok=True)

# Create __init__.py files
inits = [
    "signbridge/__init__.py", "signbridge/common/__init__.py", 
    "signbridge/data/__init__.py", "signbridge/features/__init__.py", 
    "signbridge/models/__init__.py", "signbridge/recognition/__init__.py", 
    "signbridge/generation/__init__.py", "signbridge/api/__init__.py"
]
for i in inits:
    Path(i).touch()

files = {
    "pyproject.toml": """[project]
name = "signbridge"
version = "0.1.0"
description = "Real-time, bidirectional ASL<->English translator"
readme = "README.md"
requires-python = ">=3.11"
dependencies = [
    "torch", "numpy", "pandas", "pyarrow", "scipy", "scikit-learn",
    "fastapi", "uvicorn", "pydantic", "pyyaml", "onnxruntime", "tqdm"
]

[project.optional-dependencies]
dev = ["pytest", "pytest-cov", "ruff", "mypy"]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.mypy]
strict = true
packages = ["signbridge"]

[[tool.mypy.overrides]]
module = "yaml.*"
ignore_missing_imports = true
""",

    "signbridge/common/landmarks.py": """from typing import List, Tuple

ROWS_PER_FRAME = 543

FACE: Tuple[int, int] = (0, 468)
LEFT_HAND: Tuple[int, int] = (468, 489)
POSE: Tuple[int, int] = (489, 522)
RIGHT_HAND: Tuple[int, int] = (522, 543)

HAND_CONNECTIONS: List[Tuple[int, int]] = [
    (0, 1), (1, 2), (2, 3), (3, 4),        # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),        # index
    (5, 9), (9, 10), (10, 11), (11, 12),   # middle
    (9, 13), (13, 14), (14, 15), (15, 16), # ring
    (13, 17), (17, 18), (18, 19), (19, 20),# pinky
    (0, 17)                                # palm edge
]

POSE_UPPER_BODY: List[int] = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

POSE_MIRROR_PAIRS: List[Tuple[int, int]] = [
    (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22)
]

POSE_CONNECTIONS_UPPER: List[Tuple[int, int]] = [
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16),
    (15, 17), (15, 19), (15, 21),
    (16, 18), (16, 20), (16, 22)
]

LIPS: List[int] = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308
]
""",

    "signbridge/common/config.py": """import yaml
from pathlib import Path
from typing import Type, TypeVar, Any
from pydantic import BaseModel, Field

T = TypeVar("T", bound=BaseModel)

class PathConfig(BaseModel):
    data_root: str = "data/raw"
    cache_root: str = "data/cache"
    artifacts_root: str = "artifacts"
    dictionary_root: str = "data/dictionary"

class BaseConfig(BaseModel):
    paths: PathConfig = Field(default_factory=PathConfig)
    seed: int = 42

def load_config(path: str | Path, model_cls: Type[T]) -> T:
    with open(path, "r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return model_cls.model_validate(data)
""",

    "configs/base.yaml": """paths:
  data_root: data/raw
  cache_root: data/cache
  artifacts_root: artifacts
  dictionary_root: data/dictionary
seed: 42
""",

    "Makefile": """.PHONY: setup format lint typecheck test

setup:
\tuv venv
\tuv pip install -e ".[dev]"
\tpre-commit install || echo "Consider installing pre-commit globally."

format:
\truff format signbridge tests
\truff check --select I --fix signbridge tests

lint:
\truff check signbridge tests

typecheck:
\tmypy --strict signbridge

test:
\tpytest tests/
""",

    ".pre-commit-config.yaml": """repos:
  - repo: https://github.com/astral-sh/ruff-pre-commit
    rev: v0.3.4
    hooks:
      - id: ruff
        args: [ --fix ]
      - id: ruff-format
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.9.0
    hooks:
      - id: mypy
        args: [ --strict ]
        additional_dependencies: [ pydantic ]
""",

    ".gitignore": """__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.mypy_cache/
.venv/
data/
artifacts/
""",

    "tests/test_landmarks.py": """from signbridge.common.landmarks import (
    ROWS_PER_FRAME, FACE, LEFT_HAND, POSE, RIGHT_HAND,
    HAND_CONNECTIONS, LIPS, POSE_UPPER_BODY
)

def test_slices_are_contiguous_and_cover_all() -> None:
    assert FACE[0] == 0
    assert FACE[1] == LEFT_HAND[0]
    assert LEFT_HAND[1] == POSE[0]
    assert POSE[1] == RIGHT_HAND[0]
    assert RIGHT_HAND[1] == ROWS_PER_FRAME
    
    total_len = (FACE[1] - FACE[0]) + (LEFT_HAND[1] - LEFT_HAND[0]) + \\
                (POSE[1] - POSE[0]) + (RIGHT_HAND[1] - RIGHT_HAND[0])
    
    assert total_len == ROWS_PER_FRAME
    assert ROWS_PER_FRAME == 543

def test_hand_connections_indices() -> None:
    for i, j in HAND_CONNECTIONS:
        assert 0 <= i <= 20
        assert 0 <= j <= 20

def test_lips_unique_indices() -> None:
    assert len(set(LIPS)) == 40
    for idx in LIPS:
        assert 0 <= idx < 468

def test_pose_upper_body_indices() -> None:
    for idx in POSE_UPPER_BODY:
        assert 0 <= idx <= 32
""",

    "AGENTS.md": """# Development Conventions
1. One implementation per concept. The preprocessing used in training is the same function used in live inference. Never fork it.
2. Data and feature code are pure functions over numpy arrays. torch is used only inside models/ and training.
3. Every tunable lives in configs/*.yaml and is read through config.py. No magic numbers in code.
4. Every module ships with tests. Tests run without the dataset (synthetic fixtures) unless marked @pytest.mark.dataset.
5. No notebooks in the repo. Analysis is a script under scripts/ that writes to artifacts/.
6. Report only numbers produced by running code in this session. Paste the command and its output.
7. Interfaces defined in an earlier phase are frozen. Changing one means updating every caller and test in the same change and saying so.
8. Do not add a dependency that is not in pyproject without stating why in the PR description.
9. Finish the whole phase. If something is blocked, deliver everything else and say exactly what is missing.
""",

    "docs/ARCHITECTURE.md": """# signbridge Architecture
## Pipeline 1: Recognition (ASL -> English)
*Owning Module: `recognition/` & `features/` & `models/` & `api/`*
1. **webcam** -> (TODO: phase N)
2. **landmarks** (543x3 per frame) -> (TODO: phase N)
3. **segmenter** -> (TODO: phase N)
4. **preprocess** -> (TODO: phase N)
5. **ONNX classifier** -> (TODO: phase N)
6. **word** -> (TODO: phase N)

## Pipeline 2: Generation (English -> ASL)
*Owning Module: `generation/` & `features/` & `api/`*
1. **English text** -> (TODO: phase N)
2. **gloss tokens** -> (TODO: phase N)
3. **sign dictionary lookup** -> (TODO: phase N)
4. **stitch** -> (TODO: phase N)
5. **(T,543,3) frames** -> (TODO: phase N)
6. **3D avatar** -> (TODO: phase N)
""",

    "README.md": """# signbridge
A real-time, bidirectional ASL<->English translator. 
Direction 1: webcam -> MediaPipe landmarks -> sign recognition -> English word. 
Direction 2: English text -> sequence of 3D landmark frames -> 3D avatar.
"""
}

for filepath, content in files.items():
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

print("Done initializing signbridge project files.")