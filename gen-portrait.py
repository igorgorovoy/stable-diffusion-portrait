import logging
import sys
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline

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
        # Перевіряємо залежності
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
        
        prompt = "A classical oil painting of a distinguished lady, 18th-century style, dark background, Rembrandt lighting, realistic, old canvas texture"
        negative_prompt = """deformed, distorted, disfigured, 
                           bad anatomy, changed face, different face,
                           extra limbs, extra fingers, extra features,
                           duplicate, multiple faces, blurry, 
                           bad art, cartoon, anime, sketchy"""
        
        logger.info(f"Generating image with prompt: {prompt}")
        
        # Параметри генерації
        num_inference_steps = 200  # Максимальна кількість кроків для деталізації
        guidance_scale = 9.0  # Збільшуємо для кращого дотримання промпту
        
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        output_path = "generated_portrait.png"
        image.save(output_path)
        logger.info(f"Image saved to: {output_path}")
        
        image.show()
        
    except Exception as e:
        logger.error(f"Error during image generation: {str(e)}")
        raise

if __name__ == "__main__":
    generate_portrait()
