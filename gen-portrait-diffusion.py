import logging
import sys
import torch
from diffusers import StableDiffusionPipeline
from datetime import datetime

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_dependencies():
    """Перевірка версій залежностей"""
    try:
        import numpy as np
        import transformers
        logger.info(f"NumPy version: {np.__version__}")
        logger.info(f"Torch version: {torch.__version__}")
        logger.info(f"Transformers version: {transformers.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        logger.error(f"Missing dependency: {str(e)}")
        sys.exit(1)

def generate_portrait():
    try:
        check_dependencies()
        
        logger.info("Initializing Stable Diffusion pipeline...")
        model_id = "CompVis/stable-diffusion-v1-4"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipe.to(device)
        
        prompt = """portrait of a цщьфт with mustache in classical oil painting style,
                   18th century aristocratic portrait,
                   brown jacket, white collar,
                   natural skin tones,
                   detailed facial features,
                   professional Rembrandt lighting,
                   neutral dark background,
                   museum quality painting,
                   high detail"""
        
        negative_prompt = """ugly, deformed, blurry, bad art,
                           poor quality, low quality,
                           cartoon, anime, 3d, digital art"""
        
        logger.info(f"Generating image with prompt: {prompt}")
        
        # Встановлюємо більший розмір зображення та параметри для кращої якості
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=512,
            width=512,
            num_inference_steps=100,
            guidance_scale=7.5
        ).images[0]
        
        # Додаємо timestamp до імені файлу
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"generated_portrait_diffusion_{timestamp}_1024.png"
        image.save(output_path)
        logger.info(f"Image saved to: {output_path}")
        
        image.show()
        
    except Exception as e:
        logger.error(f"Error during image generation: {str(e)}")
        raise

if __name__ == "__main__":
    generate_portrait()
