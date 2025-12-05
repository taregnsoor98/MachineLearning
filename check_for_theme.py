import re

keywords = ["flower", "flowers", "blossom", "bloom", "floral", "rose", "orchid", "tulip", "sunflower"]

def is_flower(text):
    if text is None:
        return False
    text = text.lower()
    return any(k in text for k in keywords)

filtered_indices = [i for i, item in enumerate(ds) if is_flower(item["TEXT"])]

print("Total flower matches:", len(filtered_indices))
