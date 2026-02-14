import torch
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Load pretrained BLIP model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load and preprocess image
image_path = "sample.jpg"  # Replace with your image
image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")

# Generate caption
caption_ids = model.generate(**inputs)
caption = processor.decode(caption_ids[0], skip_special_tokens=True)

print(f"Generated Caption: {caption}")
