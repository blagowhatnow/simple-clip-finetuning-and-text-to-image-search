import os
import numpy as np
import pandas as pd
import torch
import clip
from PIL import Image
from tqdm import tqdm

# Clear CUDA cache
torch.cuda.empty_cache()

# Load image paths and labels
df = pd.read_csv("data/train.tsv", sep='\t')
image_paths, labels = df['image_path'].tolist(), df['caption']
labels = [label.replace("'", "") for label in labels]

# Set device and load model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)
checkpoint = torch.load("model_checkpoint/model_epoch_4.pt", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])

# Preprocess images
preprocessed_images = []
image_paths_cleaned = []
for img_path in tqdm(image_paths, desc="Preprocessing Images"):
    try:
        image = Image.open(img_path).convert("RGB")
        preprocessed_image = preprocess(image).unsqueeze(0).to(device)
        preprocessed_images.append(preprocessed_image)
        image_paths_cleaned.append(img_path)
    except Exception as e:
        print(f"Skipping invalid image {img_path}: {e}")

# Batch encode images
batch_size = 32
image_embeddings = []
for i in range(0, len(preprocessed_images), batch_size):
    batch = torch.cat(preprocessed_images[i:i+batch_size], dim=0)
    with torch.no_grad():
        batch_embeddings = model.encode_image(batch).cpu().detach().numpy()
    image_embeddings.append(batch_embeddings)
image_embeddings = np.concatenate(image_embeddings, axis=0)

# Encode text labels
with torch.no_grad():
    label_embeddings = model.encode_text(clip.tokenize(labels).to(device)).cpu().detach().numpy()

# Save embeddings
output_dir = "embeddings"
os.makedirs(output_dir, exist_ok=True)
np.save(os.path.join(output_dir, "image_embeddings.npy"), image_embeddings, allow_pickle=True)
np.save(os.path.join(output_dir, "label_embeddings.npy"), label_embeddings, allow_pickle=True)

print(f"Processed {len(preprocessed_images)} valid images out of {len(image_paths)}.")
print("Embeddings saved successfully!")

