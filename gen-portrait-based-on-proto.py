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

def prepare_image(image_path, target_size=(512, 512)):
    """Підготовка зображення до потрібного формату"""
    image = Image.open(image_path)
    # Конвертуємо в RGB якщо потрібно
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # Змінюємо розмір зі збереженням пропорцій
    image.thumbnail(target_size, Image.Resampling.LANCZOS)
    # Створюємо нове зображення з білим фоном
    new_image = Image.new('RGB', target_size, (255, 255, 255))
    # Вставляємо оригінальне зображення по центру
    offset = ((target_size[0] - image.size[0]) // 2,
              (target_size[1] - image.size[1]) // 2)
    new_image.paste(image, offset)
    return new_image

def generate_portrait_from_references():
    try:
        check_dependencies()
        
        # Завантажуємо та підготовлюємо зображення
        logger.info("Loading and preparing reference images...")
        my_image = prepare_image("iam.png")
        belamy_image = prepare_image("Edmond_de_Belamy.png")
        
        # Створюємо композитне зображення як основу
        composite = Image.blend(my_image, belamy_image, 0.8)
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
        
        prompt = """A masterful oil painting in the style of Edmond de Belamy, 
                   portrait in classical 18th-century style, 
                   dark mysterious background, Rembrandt lighting, 
                   realistic details, old canvas texture, 
                   elegant and sophisticated atmosphere"""
        
        negative_prompt = "cartoon, anime, sketchy, double image, duplicate, deformed"
        
        logger.info(f"Generating image with prompt: {prompt}")
        
        # Параметри генерації
        strength = 0.65  # Трохи менше, щоб зберегти більше деталей оригіналу
        guidance_scale = 8.5  # Трохи вище для кращого дотримання стилю
        num_inference_steps = 100  # Більше кроків для кращої якості
        
        # Генеруємо зображення
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=composite,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps
        ).images[0]
        
        output_path = "generated_belamy_portrait.png"
        image.save(output_path)
        logger.info(f"Image saved to: {output_path}")
        
        image.show()
        
    except Exception as e:
        logger.error(f"Error during image generation: {str(e)}")
        raise

if __name__ == "__main__":
    generate_portrait_from_references()
