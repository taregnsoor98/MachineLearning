import mport torch
import os
import json
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from diffusers import StableDiffusionPipeline, DDPMScheduler
from peft import LoraConfig, get_peft_model
import torchvision.transforms as transforms
from tqdm.auto import tqdm
import datetime

# ====================
# SETTINGS
# ====================
model_id = "runwayml/stable-diffusion-v1-5"
train_dir = "/content/dataset/flower_dataset/train"
save_dir = "/content/flower_lora_15k_full"
batch_size = 2
lr = 1e-4
epochs = 1
max_images = 15000

# Log file
log_file = "/content/training_log.json"
loss_log = []

# ====================
# DATASET
# ====================
class FlowerDataset(Dataset):
    def __init__(self, folder, max_images=None):
        self.files = sorted([f for f in os.listdir(folder) if f.endswith(".jpg")])
        if max_images:
            self.files = self.files[:max_images]
        self.folder = folder
        self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, self.files[idx])
        txt_path = img_path.replace(".jpg", ".txt")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read().strip()

        return image, text

# ====================
# TRAINING
# ====================
def main():
    print("FULL TRAINING - 15,000 IMAGES")
    print(f"Log file: {log_file}")
    print(f"Estimated: 2-3 hours")
    print("=" * 50)

    # Start log
    loss_log.append({
        "start_time": datetime.datetime.now().isoformat(),
        "total_images": max_images,
        "batch_size": batch_size,
        "total_steps": max_images // batch_size
    })

    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")

    pipe.enable_xformers_memory_efficient_attention()

    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    unet = pipe.unet
    vae = pipe.vae
    noise_scheduler = DDPMScheduler.from_config(pipe.scheduler.config)

    # Freeze models
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)

    # Setup LoRA
    print("Applying LoRA...")
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["to_q", "to_v", "to_k", "to_out.0"],
        lora_dropout=0.05,
        bias="none",
    )
    unet = get_peft_model(unet, lora_config)
    unet.train()

    # Create dataset
    print("Loading dataset...")
    dataset = FlowerDataset(train_dir, max_images=max_images)
    print(f"   Training on ALL {len(dataset)} images")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)

    # Training loop
    print(f"Starting FULL training...")
    global_step = 0

    for epoch in range(epochs):
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}")

        for images, texts in progress_bar:
            # Move to GPU
            images = images.to("cuda", dtype=torch.float16)

            # Tokenize text
            inputs = tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True
            ).to("cuda")

            # Convert images to latents
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample() * 0.18215

            # Add noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (latents.shape[0],), device="cuda")
            noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

            # Get text embeddings
            with torch.no_grad():
                encoder_hidden_states = text_encoder(inputs.input_ids).last_hidden_state

            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

            # Calculate loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)

            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Log loss every 100 steps
            if global_step % 100 == 0:
                loss_log.append({
                    "step": global_step,
                    "loss": float(loss.item()),
                    "timestamp": datetime.datetime.now().isoformat()
                })
                # Print to console
                print(f"Step {global_step}, Loss: {loss.item():.4f}")
                # Save to file
                with open(log_file, 'w') as f:
                    json.dump(loss_log, f, indent=2)

            # Update progress bar
            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}", "Step": global_step})
            global_step += 1

            # Save checkpoint every 2000 steps
            if global_step % 2000 == 0:
                checkpoint_dir = f"{save_dir}_checkpoint_step_{global_step}"
                os.makedirs(checkpoint_dir, exist_ok=True)
                unet.save_pretrained(checkpoint_dir)
                print(f"Checkpoint saved: {checkpoint_dir}")

    # Final save
    print("Saving final model...")
    os.makedirs(save_dir, exist_ok=True)
    unet.save_pretrained(save_dir)

    # Final log entry
    loss_log.append({
        "end_time": datetime.datetime.now().isoformat(),
        "total_steps_completed": global_step,
        "final_loss": float(loss.item())
    })

    with open(log_file, 'w') as f:
        json.dump(loss_log, f, indent=2)

    print(f"FULL training complete!")
    print(f"Total steps: {global_step}")
    print(f"Loss log saved to: {log_file}")
    print(f"Model saved to: {save_dir}")

if __name__ == "__main__":
    main()