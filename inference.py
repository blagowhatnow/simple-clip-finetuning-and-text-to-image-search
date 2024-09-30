import skimage
from skimage import io
import clip
import torch
import os
from PIL import Image
import numpy as np
import pandas as pd

# Clear CUDA cache
torch.cuda.empty_cache()

# Load images and labels
df=pd.read_csv("data/train.tsv", sep='\t')
image_paths, labels = df['image_path'].tolist(), df['caption']
labels = [i.replace("'", "") for i in labels]  # Clean up labels if necessary

# Set device for the model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)  # Must set jit=False for training
# Load the model checkpoint (if needed)
checkpoint = torch.load("model_checkpoint/model_epoch_4.pt")
model.load_state_dict(checkpoint['model_state_dict'])


# Preprocess images and encode them
preprocessed_images = []

for img_path in image_paths:
    try:
        image = Image.open(img_path).convert("RGB")  # Ensure image is in RGB format
        preprocessed_image = preprocess(image).unsqueeze(0).to(device)  # Preprocess and add batch dimension
        preprocessed_images.append(preprocessed_image)
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")

# Stack all the preprocessed images into a single tensor
preprocessed_images = torch.cat(preprocessed_images, dim=0)

# Encode images and labels
with torch.no_grad():
    image_embeddings = model.encode_image(preprocessed_images).cpu().detach().numpy()  # Get image embeddings
    label_embeddings = model.encode_text(clip.tokenize(labels).to(device)).cpu().detach().numpy()  # Get label embeddings

# Save embeddings to .npy files
with open('image_embeddings.npy', 'wb') as f:
    np.save(f, image_embeddings, allow_pickle=True)

with open('label_embeddings.npy', 'wb') as f:
    np.save(f, label_embeddings, allow_pickle=True)

print("Embeddings saved successfully!")

