import logging
import sys
import torch
from PIL import Image
import numpy as np
from diffusers import StableDiffusionImg2ImgPipeline

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

def prepare_image(image_path, target_size=(1024, 1024)):
    """Підготовка зображення до потрібного формату"""
    image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    new_image = Image.new('RGB', target_size, (255, 255, 255))
    offset = ((target_size[0] - image.size[0]) // 2,
              (target_size[1] - image.size[1]) // 2)
    new_image.paste(image, offset)
    return new_image

def generate_portrait_from_references():
    try:
        check_dependencies()
        
        logger.info("Loading and preparing reference images...")
        target_size = (1024, 1024)
        my_image = prepare_image("iam.png", target_size)
        belamy_image = prepare_image("Edmond_de_Belamy.png", target_size)
        
        # Змінюємо баланс змішування
        composite = Image.blend(my_image, belamy_image, 0.15)  # Ще менше впливу картини Беламі
        composite.save("composite_reference.png")
        logger.info("Created composite reference image")
        
        logger.info("Initializing Stable Diffusion img2img pipeline...")
        model_id = "CompVis/stable-diffusion-v1-4"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )
        pipe.to(device)
        
        # Оновлений промпт з урахуванням стилю зображення
        prompt = """portrait of a man with mustache,
                   realistic oil painting,
                   classical portrait style,
                   brown jacket, white collar,
                   natural skin tones,
                   detailed facial features,
                   professional lighting,
                   neutral background,
                   high quality painting"""
        
        # Спростимо негативний промпт
        negative_prompt = """ugly, deformed, blurry, bad art,
                           poor quality, low quality,
                           extra features, double image"""
        
        # Налаштування параметрів
        strength = 0.35  # Зменшуємо силу трансформації
        guidance_scale = 7.5  # Середнє значення для балансу
        num_inference_steps = 75  # Оптимальна кількість кроків
        
        logger.info("Starting image generation...")
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=composite,
            strength=strength,  # Менша сила трансформації
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=512,  # Зменшуємо розмір для тестування
            width=512
        ).images[0]
        
        output_path = "generated_portrait_blend_1024.png"
        image.save(output_path)
        logger.info(f"Image saved to: {output_path}")
        
        image.show()
        
    except Exception as e:
        logger.error(f"Error during image generation: {str(e)}")
        raise

if __name__ == "__main__":
    generate_portrait_from_references()
