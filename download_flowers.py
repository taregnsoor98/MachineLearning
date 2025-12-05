import os
import json
import requests
from PIL import Image
from io import BytesIO

# Folder to save downloaded images + captions
save_dir = r"C:\Users\pc\Desktop\txt2img\text2image_project\flower_dataset"
os.makedirs(save_dir, exist_ok=True)

# Load the 20,000 filtered flower metadata
with open("flower_subset_metadata.json", "r", encoding="utf-8") as f:
    flower_subset = json.load(f)

print(f"Loaded {len(flower_subset)} flower samples.")

# Start downloading
for i, item in enumerate(flower_subset):
    url = item["URL"]
    caption = item["TEXT"]

    try:
        response = requests.get(url, timeout=8)
        img = Image.open(BytesIO(response.content)).convert("RGB")

        # Save image
        img.save(f"{save_dir}/{i}.jpg", "JPEG")

        # Save caption
        with open(f"{save_dir}/{i}.txt", "w", encoding="utf-8") as f_caption:
            f_caption.write(caption)

        print(f"Downloaded {i+1} / {len(flower_subset)}")

    except Exception as e:
        print(f"Failed {i}: {e}")
        continue

print("Done downloading flower dataset!")
