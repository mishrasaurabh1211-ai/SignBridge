from pathlib import Path

# 1. Fix unused variable in test
p = Path("tests/test_features.py")
txt = p.read_text("utf-8")
txt = txt.replace("x_res2, mask2 = resample", "_, mask2 = resample")
p.write_text(txt, "utf-8")

# 2. Add type arguments to dict in preprocess.py
p = Path("signbridge/features/preprocess.py")
txt = p.read_text("utf-8")
txt = txt.replace("from dataclasses import dataclass", "from dataclasses import dataclass\nfrom typing import Any")
txt = txt.replace("meta: dict", "meta: dict[str, Any]")
p.write_text(txt, "utf-8")

# 3. Add type arguments to Dataset in dataset.py
p = Path("signbridge/features/dataset.py")
txt = p.read_text("utf-8")
txt = txt.replace("class SignDataset(Dataset):", "class SignDataset(Dataset[dict[str, torch.Tensor]]):")
p.write_text(txt, "utf-8")

print("Type hints patched successfully.")