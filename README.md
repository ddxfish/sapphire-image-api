# SDXL API

Image generation backend for [Sapphire](https://github.com/ddxfish/sapphire). Flask REST API that loads an SDXL model once and serves fast inference. Works on Linux, may work on Windows. Requires Nvidia.

**Requires NVIDIA GPU with CUDA.**

## VRAM Requirements

- **~8.5 GB idle** after model load
- **~12-15 GB peak** during generation (varies by scheduler, euler_a is lower VRAM)
- **FP16** (default): RTX 20-series or newer
- **FP32**: GTX 10-series and older, doubles VRAM requirements

## Quickstart

```bash
# Setup environment
conda create -n sdxl python=3.11 -y
conda activate sdxl
pip install -r requirements.txt

# Add your model
mkdir models
# Drop an SDXL .safetensors file into models/

# Run
python main.py
```

Server starts on `http://localhost:5153` and has no visible UI. 

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/generate` | POST | Start generation (returns immediately) |
| `/image-status` | GET | Poll generation status |
| `/latest-image` | GET | Get latest image |
| `/output/{id}.jpg` | GET | Get specific image |
| `/status` | GET | Server health |

## Generate Request

```json
{
  "prompt": "a cat in space",
  "negative_prompt": "blurry",
  "height": 1024,
  "width": 1024,
  "steps": 20,
  "guidance_scale": 7.5,
  "scale": 1.0,
  "scheduler": "euler_a"
}
```

Schedulers: `default`, `ddim`, `euler`, `euler_a`, `dpm++`, `unipc`, `dpm++_sde_karras`, `dpm++_2m_karras`