🌉 SignBridge: Real-Time Bidirectional ASL ↔ English Translator

A production-grade machine learning pipeline for real-time American Sign Language (ASL) recognition and generation, engineered with a focus on strict modularity, high-performance data pipelines, and training-inference parity.

🚀 Project Overview

SignBridge is a comprehensive system designed to bridge the communication gap between ASL signers and English speakers.

Unlike standard research repositories, this project is architected for production readiness. It enforces strict typing (mypy), aggressive linting (ruff), and utilizes a highly optimized data engineering layer capable of processing massive datasets without I/O bottlenecks. The core design philosophy ensures that the exact same feature extraction logic runs during model training and live webcam inference.

✨ Key Engineering Features

High-Performance Data Ingestion: Processes a 12.7+ GB dataset (94,500+ sequences) using pyarrow and caches it into contiguous NumPy memory-maps (memmap). This eliminates I/O bottlenecks, ensuring GPU saturation during training.

Training-Inference Parity: A stateless, purely functional feature extraction pipeline guarantees that the model sees the exact same tensor format during real-time webcam inference as it did during batch training.

Strict Type Safety & Linting: Built with Python 3.11 features, completely validated by mypy --strict and ruff, ensuring robust, crash-free execution.

Advanced Cross-Validation: Utilizes GroupKFold splitting on participant IDs to guarantee zero data leakage between training and validation sets.

Configuration-Driven: All hyper-parameters, network architectures, and feature toggles (e.g., enabling/disabling facial landmarks or motion vectors) are managed via centralized YAML configurations and Pydantic models.

🏗️ Architecture

1. Data Layer (signbridge.data)

Handles raw data ingestion from Google's Isolated Sign Language Recognition dataset.

Reads raw Parquet files (543 landmarks per frame) via PyArrow.

Validates dataset invariants (exact row counts, label mappings).

Caches sequences into fast float16 NumPy arrays and builds metadata indices for lightning-fast dataloading.

2. Feature Pipeline (signbridge.features)

Transforms raw landmark coordinates into normalized, model-ready tensors.

Trimming: Dynamically removes leading/trailing frames lacking hand detections.

Global Spatial Normalization: Body-centric normalization (origin at shoulder midpoint, scaled by shoulder width) invariant to camera distance.

Handedness Canonicalization: Dynamically detects the dominant hand and mirrors left-handed signers to standard right-handed representations.

Motion & Shape Features: Calculates time-derivatives (motion vectors) and bounding-box-normalized hand shapes.

Resampling: Linearly interpolates or zero-pads all sequences to a fixed temporal length (64 frames).

🛠️ Tech Stack

Core: Python 3.11, PyTorch

Data Engineering: PyArrow, Pandas, NumPy, Scikit-learn

Package Management: uv (lightning-fast Python package manager)

Code Quality: ruff, mypy, pytest

Serving (Planned): FastAPI, ONNX Runtime

🏁 Getting Started

Prerequisites

Python 3.11+

uv package manager

Installation

Clone the repository:

git clone https://github.com/YOUR_USERNAME/SignBridge.git
cd SignBridge


Create a virtual environment and install dependencies using uv:

uv venv
# On Windows: .venv\Scripts\activate
# On macOS/Linux: source .venv/bin/activate
uv pip install -e ".[dev]"


Run the test suite to verify the environment:

pytest tests/


Check code quality:

ruff check signbridge tests
mypy --strict signbridge
