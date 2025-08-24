# Models Prediction Project

A Python project that demonstrates multimodal feature processing using both local ML models and Ollama API integration.

## Overview

This project contains two main scripts for processing different types of features (images, text, integers, and floats) using different approaches:

1. **`multi-feature-updated.py`** - Local ML models using PyTorch and Transformers
2. **`multi_features_ollama_qwen2_5vl.py`** - Cloud-based processing using Ollama API with Qwen2.5-VL model

## Features

### Supported Feature Types

- **Image**: Base64-encoded images for object detection or captioning
- **String**: Text processing with BERT fill-mask or general responses
- **Integer**: Simple mathematical transformations (e.g., x * 2)
- **Float**: Mathematical operations (e.g., round(x², 1))

### Script 1: Local ML Models (`multi-feature-updated.py`)

Uses local PyTorch models for processing:
- **Vision**: Faster R-CNN object detection via torchvision
- **Text**: BERT "fill-mask" pipeline via transformers
- **Math**: Simple arithmetic operations

#### Capabilities
- Object detection with bounding boxes and confidence scores
- Text completion for masked language modeling
- Basic mathematical transformations
- GPU acceleration support (CUDA)

### Script 2: Ollama API Integration (`multi_features_ollama_qwen2_5vl.py`)

Uses Ollama's Qwen2.5-VL model for cloud-based processing:
- **Vision**: Object detection and image captioning
- **Text**: Natural language understanding and generation
- **Math**: Mathematical reasoning and computation

#### Capabilities
- Multi-modal understanding (text + images)
- JSON-structured outputs
- Configurable via environment variables

## Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

### Environment Variables (for Ollama script)

Set these environment variables for the Ollama integration:

```bash
# Ollama server URL (default: http://localhost:11434/api/chat)
export OLLAMA_URL="http://localhost:11434/api/chat"

# Model name (default: qwen2.5vl)
export OLLAMA_MODEL="qwen2.5vl"
```

## Usage

### Local ML Models

```bash
python multi-feature-updated.py
```

**Example Output:**
```
Details: {'model': 'MultiModalModel', 'description': 'Unified model handling typed images, strings (BERT mask), ints, and floats.', 'capabilities': {'image': True, 'string_fill_mask': True, 'int': True, 'float': True}}
[0] image -> 3 detections > 0.5
  label_id=5, score=0.98, box=[0.0, 0.0, 640.0, 480.0]
[1] int -> 14
[2] float -> 6.2
[3] string -> [{'sequence': 'the capital of france is paris.', 'score': 0.123, 'token': 2054, 'token_str': 'paris'}]
```

### Ollama API Integration

```bash
python multi_features_ollama_qwen2_5vl.py
```

**Example Output:**
```json
{
  "0": {
    "labels": ["bus", "vehicle"],
    "boxes": [[10, 20, 200, 150]]
  },
  "1": {"y": 14},
  "2": {"y": 6.2},
  "3": {"answer": "paris"},
  "4": {"answer": "hello"}
}
```

## Input Format

Both scripts expect a list of typed features:

```python
features = [
    {"type": "image",  "value": "base64_encoded_image_string"},
    {"type": "int",    "value": 7},
    {"type": "float",  "value": 2.5},
    {"type": "string", "value": "The capital of France is [MASK]."}
]
```

## File Structure

```
ModelsPrediction/
├── multi-feature-updated.py          # Local ML models
├── multi_features_ollama_qwen2_5vl.py # Ollama API integration
├── requirements.txt                   # Python dependencies
├── readme.md                         # This file
└── bus.jpg                           # Sample image for testing
```

## Requirements

### Core Dependencies
- `requests>=2.25.0` - HTTP client for Ollama API

### Local ML Dependencies (for multi-feature-updated.py)
- `transformers>=4.30.0` - Hugging Face transformers
- `torch>=2.0.0` - PyTorch
- `torchvision>=0.15.0` - Computer vision models
- `pillow>=9.0.0` - Image processing
- `numpy>=1.21.0` - Numerical computing

## Notes

- The local ML script requires significant disk space for model downloads
- Ollama script requires a running Ollama server with Qwen2.5-VL model
- Image inputs must be base64-encoded strings (no data URL prefix)
- Both scripts preserve input order in their outputs

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use CPU mode
2. **Model download fails**: Check internet connection and disk space
3. **Ollama connection error**: Verify Ollama server is running and accessible
4. **Image processing error**: Ensure images are valid base64-encoded strings

### Performance Tips

- Use GPU acceleration when available for local models
- Batch multiple requests when possible
- Consider model quantization for memory-constrained environments
