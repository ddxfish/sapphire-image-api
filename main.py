#!/usr/bin/env python3
"""
SDXL Flask API Server
Loads an SDXL model once, serves fast image generation via REST API.
Designed for local GPU inference with CUDA.
"""

import os
import io
import gc
import base64
import random
import hashlib
import logging
import argparse
import threading
from datetime import datetime

import torch
from PIL import Image
from flask import Flask, request, send_file, jsonify
from diffusers import StableDiffusionXLPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global state
pipe = None
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

latest_image = {
    'data': None,
    'timestamp': None,
    'prompt': None,
    'seed': None,
    'generating': False,
    'image_count': 0,
    'current_id': None
}


def generate_unique_id():
    """Generate unique ID from timestamp and random hash."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    random_hash = hashlib.md5(str(random.random()).encode()).hexdigest()[:8]
    return f"{timestamp}_{random_hash}"


def load_model(model_path, use_fp16=True):
    """Load the Stable Diffusion XL model."""
    global pipe

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This API requires a CUDA-capable GPU.")

    logger.info(f"Loading model: {model_path}")
    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")

    torch_dtype = torch.float16 if use_fp16 else torch.float32
    logger.info(f"Precision: {'FP16' if use_fp16 else 'FP32'}")

    try:
        pipe = StableDiffusionXLPipeline.from_single_file(model_path, torch_dtype=torch_dtype)
        pipe = pipe.to("cuda")

        # Enable memory optimizations
        try:
            pipe.enable_xformers_memory_efficient_attention()
            logger.info("Enabled xformers attention")
        except Exception:
            logger.warning("xformers not available, using default attention")

        try:
            pipe.enable_vae_slicing()
            logger.info("Enabled VAE slicing")
        except Exception:
            pass

        torch.cuda.empty_cache()
        gc.collect()

        logger.info("Model loaded successfully")
        return pipe

    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise


def generate_image_thread(unique_id, prompt, height, width, steps, guidance_scale, 
                          negative_prompt, scale, scheduler, seed):
    """Generate image in background thread."""
    global latest_image, pipe

    try:
        latest_image['generating'] = True
        latest_image['prompt'] = prompt
        latest_image['seed'] = seed
        latest_image['current_id'] = unique_id

        generator = torch.Generator(device='cuda').manual_seed(seed)

        # Swap scheduler if requested
        original_scheduler = None
        if scheduler != "default":
            original_scheduler = pipe.scheduler
            try:
                from diffusers import (
                    DDIMScheduler,
                    EulerDiscreteScheduler,
                    EulerAncestralDiscreteScheduler,
                    DPMSolverMultistepScheduler,
                    UniPCMultistepScheduler,
                    DPMSolverSinglestepScheduler,
                    KDPM2DiscreteScheduler
                )

                schedulers = {
                    "ddim": DDIMScheduler,
                    "euler": EulerDiscreteScheduler,
                    "euler_a": EulerAncestralDiscreteScheduler,
                    "dpm++": DPMSolverMultistepScheduler,
                    "unipc": UniPCMultistepScheduler,
                }

                if scheduler in schedulers:
                    pipe.scheduler = schedulers[scheduler].from_config(pipe.scheduler.config)
                elif scheduler == "dpm++_sde_karras":
                    pipe.scheduler = DPMSolverSinglestepScheduler.from_config(
                        pipe.scheduler.config, algorithm_type="sde-dpmsolver++", use_karras_sigmas=True
                    )
                elif scheduler == "dpm++_2m_karras":
                    pipe.scheduler = KDPM2DiscreteScheduler.from_config(
                        pipe.scheduler.config, use_karras_sigmas=True
                    )

                logger.info(f"Using scheduler: {scheduler}")
            except Exception as e:
                logger.warning(f"Failed to set scheduler '{scheduler}': {e}")
                original_scheduler = None

        # Generate
        with torch.no_grad():
            image = pipe(
                prompt,
                height=height,
                width=width,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                negative_prompt=negative_prompt,
                generator=generator
            ).images[0]

        # Restore original scheduler
        if original_scheduler is not None:
            pipe.scheduler = original_scheduler

        # Scale if requested
        if scale != 1.0:
            new_size = (int(image.width * scale), int(image.height * scale))
            image = image.resize(new_size, Image.LANCZOS)

        # Convert to base64
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=95)
        img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Save files
        unique_path = os.path.join(output_dir, f"{unique_id}.jpg")
        latest_path = os.path.join(output_dir, "latest-image.jpg")
        image.save(unique_path, format="JPEG", quality=95)
        image.save(latest_path, format="JPEG", quality=95)

        # Update state
        latest_image['data'] = img_str
        latest_image['timestamp'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        latest_image['image_count'] += 1
        latest_image['generating'] = False

        torch.cuda.empty_cache()
        logger.info(f"Generated [{unique_id}]: {prompt[:50]}...")

    except Exception as e:
        logger.error(f"Generation failed: {e}")
        latest_image['generating'] = False


@app.route('/generate', methods=['POST'])
def generate_image_api():
    """Start image generation. Returns immediately with image_id for polling."""
    if pipe is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "JSON body required"}), 400

        required = ['prompt', 'height', 'width', 'steps', 'guidance_scale', 
                    'negative_prompt', 'scale', 'scheduler']
        missing = [p for p in required if p not in data]
        if missing:
            return jsonify({"error": f"Missing parameters: {', '.join(missing)}"}), 400

        if not data['prompt'].strip():
            return jsonify({"error": "Prompt cannot be empty"}), 400

        unique_id = generate_unique_id()
        seed = data.get('seed', random.randint(0, 2**32 - 1))

        threading.Thread(target=generate_image_thread, kwargs={
            'unique_id': unique_id,
            'prompt': data['prompt'],
            'height': int(data['height']),
            'width': int(data['width']),
            'steps': int(data['steps']),
            'guidance_scale': float(data['guidance_scale']),
            'negative_prompt': data['negative_prompt'],
            'scale': float(data['scale']),
            'scheduler': data['scheduler'],
            'seed': seed
        }).start()

        return jsonify({
            "status": "generating",
            "message": "Image generation started",
            "seed": seed,
            "image_id": unique_id
        })

    except Exception as e:
        logger.error(f"Generate endpoint error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/output/<filename>')
def serve_output_file(filename):
    """Serve image file from output directory."""
    try:
        # Sanitize filename
        if '..' in filename or filename.startswith('/'):
            return jsonify({"error": "Invalid filename"}), 400

        file_path = os.path.join(output_dir, filename)
        if not os.path.exists(file_path):
            return jsonify({"error": "Image not found"}), 404

        return send_file(file_path, mimetype='image/jpeg')
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/latest-image', methods=['GET'])
def get_latest_image():
    """Get the most recently generated image."""
    if latest_image['data'] is None:
        return jsonify({"error": "No image generated yet"}), 404

    img_data = base64.b64decode(latest_image['data'])
    img_byte_arr = io.BytesIO(img_data)
    img_byte_arr.seek(0)

    response = send_file(img_byte_arr, mimetype='image/jpeg')
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route('/image-status', methods=['GET'])
def get_image_status():
    """Poll generation status."""
    return jsonify({
        "has_image": latest_image['data'] is not None,
        "timestamp": latest_image['timestamp'],
        "prompt": latest_image['prompt'],
        "seed": latest_image['seed'],
        "generating": latest_image['generating'],
        "image_count": latest_image['image_count'],
        "image_id": latest_image['current_id']
    })


@app.route('/status', methods=['GET'])
def status():
    """Server health check."""
    if pipe is None:
        return jsonify({"status": "error", "message": "Model not loaded"}), 500

    cuda_info = {
        "device_name": torch.cuda.get_device_name(0),
        "memory_allocated": f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB",
        "memory_reserved": f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB",
    }

    return jsonify({
        "status": "ready",
        "device": str(pipe.device) if hasattr(pipe, 'device') else "cuda",
        "cuda_info": cuda_info,
        "images_generated": latest_image['image_count']
    })


@app.route('/', methods=['GET'])
def root():
    """API info."""
    return jsonify({
        "name": "SDXL API",
        "status": "ready" if pipe is not None else "model not loaded",
        "endpoints": {
            "POST /generate": "Start image generation",
            "GET /image-status": "Poll generation status",
            "GET /latest-image": "Get latest image file",
            "GET /output/<filename>": "Get specific image",
            "GET /status": "Server health"
        }
    })


def discover_models(models_dir='models'):
    """Find .safetensors files in models directory."""
    if not os.path.exists(models_dir):
        return []
    return [f for f in os.listdir(models_dir) if f.endswith('.safetensors')]


def main():
    parser = argparse.ArgumentParser(description='SDXL API Server')
    parser.add_argument('--model', type=str, default=os.environ.get('SDXL_MODEL_PATH'),
                        help='Path to SDXL .safetensors model (or set SDXL_MODEL_PATH env var)')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=5152, help='Server port')
    parser.add_argument('--fp16', action='store_true', default=True, help='Use FP16 (default)')
    parser.add_argument('--fp32', action='store_true', help='Use FP32 instead of FP16')

    args = parser.parse_args()

    # Model discovery: --model > env var > auto-detect from models/
    model_path = args.model
    if not model_path:
        found = discover_models()
        if len(found) == 1:
            model_path = os.path.join('models', found[0])
            logger.info(f"Auto-detected model: {model_path}")
        elif len(found) > 1:
            logger.error(f"Multiple models found in models/: {', '.join(found)}")
            logger.error("Specify which to use with --model models/<filename>")
            return 1
        else:
            parser.error("No model found. Drop a .safetensors in models/ or use --model")

    use_fp16 = not args.fp32

    try:
        load_model(model_path, use_fp16=use_fp16)
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        return 1

    logger.info(f"Starting API server on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()