from diffusers import StableDiffusionPipeline
import torch
import os

# Create output folder
output_dir = "before_training_outputs"
os.makedirs(output_dir, exist_ok=True)

# Load Stable Diffusion 1.5 (base model, before training)
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
pipe = pipe.to("cpu")

# Your 12 prompts (art, food, literal flowers)
prompts = [
    # ART
    "A watercolor painting of flowers in pastel colors.",
    "A detailed digital illustration of roses with vibrant lighting.",
    "A traditional oil painting of a flower bouquet on a table.",
    "A fantasy art scene with glowing magical flowers.",

    # FOOD
    "A cake decorated with edible flowers on a wooden table.",
    "A cup of tea in a floral-patterned teacup with matching saucer.",
    "A gourmet dessert topped with fresh blossoms.",
    "A fancy café plate with floral decorations and pastries.",

    # LITERAL FLOWERS
    "A close-up macro photograph of a blooming pink flower.",
    "A field of colorful flowers under soft morning light.",
    "A single rose in dramatic lighting with a dark background.",
    "A bouquet of wildflowers in a glass vase."
]

# Generate and save each image
for i, prompt in enumerate(prompts):
    print(f"Generating image {i+1} / {len(prompts)}...")
    image = pipe(prompt).images[0]
    image_path = os.path.join(output_dir, f"before_image_{i+1}.png")
    image.save(image_path)
    print(f"Saved: {image_path}")

print("All before-training images generated!")
