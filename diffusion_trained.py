from diffusers import StableDiffusionImg2ImgPipeline
import torch
from PIL import Image

pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

def dummy_safety_checker(images, clip_input):
    if isinstance(images, list):
        batch_size = len(images)
    else:
        batch_size = images.shape[0]
    return images, [False] * batch_size

pipe.safety_checker = dummy_safety_checker

def run_img2img(input_path, output_path, prompt):
    img = Image.open(input_path).convert("RGB").resize((512, 512))

    result = pipe(
        prompt=prompt,
        image=img,
        strength=0.25,
        guidance_scale=5.5,
        num_inference_steps=1000,
    )

    out = result.images[0]
    out.save(output_path)

prompt = (
    "A high-resolution aerial orthophoto, smooth transition,realistic textures, natural colors, "
    "consistent lighting, satellite imagery style."
)
run_img2img(
    "./images_patched/75-2021-0655-6870-LA93-0M20-E080-1837_jpeg_jpg.rf.9af2fe92a816e32ebb3aa31e25a43bcd_round0_bg2814_patched.jpg",
    "./generated_img.jpg",
    prompt
)