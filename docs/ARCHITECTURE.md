# signbridge Architecture
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
