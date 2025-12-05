from diffusers import StableDiffusionPipeline
import torch
import os

# Output folder
output_dir = "after_training_outputs"
os.makedirs(output_dir, exist_ok=True)

# Base model on CPU
model_id = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float32
).to("cpu")

# Load LoRA weights
lora_path = "./flower_lora_15k"
pipe.load_lora_weights(lora_path)

# Prompts
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

# Generate
for i, prompt in enumerate(prompts):
    print(f"Generating image {i+1}/{len(prompts)}...")
    image = pipe(prompt).images[0]
    path = os.path.join(output_dir, f"after_image_{i+1}.png")
    image.save(path)
    print("Saved:", path)

print("Done!")
