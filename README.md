# SDXL API

Flask API server for SDXL image generation. Loads model once, serves fast inference via REST.

**Requires CUDA GPU.**

## Setup

```bash
pip install -r requirements.txt
mkdir models
```

Drop your `.safetensors` model into `models/` - it will be auto-detected on startup.

## Usage

```bash
# Auto-detect single model in models/
python sdxl_api.py

# Multiple models? Specify which one
python sdxl_api.py --model models/your_model.safetensors

# Or use environment variable
export SDXL_MODEL_PATH=/path/to/model.safetensors
python sdxl_api.py

# Custom port
python sdxl_api.py --port 5152
```

## API

**POST /generate** - Start generation (async)
```json
{
  "prompt": "a cat in space",
  "negative_prompt": "blurry, bad quality",
  "height": 1024,
  "width": 1024,
  "steps": 20,
  "guidance_scale": 7.5,
  "scale": 1.0,
  "scheduler": "default",
  "seed": 12345
}
```
Returns `image_id` immediately. Poll `/image-status` for completion.

**GET /image-status** - Check generation status  
**GET /latest-image** - Get latest generated image  
**GET /output/{image_id}.jpg** - Get specific image  
**GET /status** - Server health  

## Schedulers

`default`, `ddim`, `euler`, `euler_a`, `dpm++`, `unipc`, `dpm++_sde_karras`, `dpm++_2m_karras`

## Test with curl

```bash
# Check server status
curl http://localhost:5152/status

# Generate image
curl -X POST http://localhost:5152/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "a cat astronaut floating in space, digital art",
    "negative_prompt": "blurry, low quality, distorted",
    "height": 1024,
    "width": 1024,
    "steps": 20,
    "guidance_scale": 7.5,
    "scale": 1.0,
    "scheduler": "euler_a"
  }'

# Poll until generating=false
curl http://localhost:5152/image-status

# Download the image
curl http://localhost:5152/latest-image -o generated.jpg
```