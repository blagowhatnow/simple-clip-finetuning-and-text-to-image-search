import os
import clip
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import ToTensor
from skimage import io
import pandas as pd

# Ensure the checkpoint directory exists
os.makedirs('model_checkpoint', exist_ok=True)

# Load additional image paths from CSV file
df = pd.read_csv('data/train.tsv', sep='\t')
image_paths = df['image_path'].tolist()
labels=df['caption']

# Load the CLIP model
device = "cuda:0" if torch.cuda.is_available() else "cpu"  # Use GPU if available
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)  # Must set jit=False for training

class ImageTitleDataset(Dataset):
    def __init__(self, list_image_path, list_txt):
        self.image_path = list_image_path
        self.title = clip.tokenize(list_txt)  # Tokenize everything at once (slow at the beginning)

    def __len__(self):
        return len(self.title)

    def __getitem__(self, idx):
        image = preprocess(Image.open(self.image_path[idx]))  # Load and preprocess image
        title = self.title[idx]
        return image, title

# Use your own data
list_txt = labels
dataset = ImageTitleDataset(image_paths, list_txt)

# Set a larger batch size
train_dataloader = DataLoader(dataset, batch_size=8, shuffle=True)  # Batch size > 1 for training

# Helper function to convert model to fp32
def convert_models_to_fp32(model): 
    for p in model.parameters(): 
        p.data = p.data.float() 
        if p.grad is not None:  # Check if gradients exist before converting
            p.grad.data = p.grad.data.float()

# Handle model precision conversion based on device
if device == "cpu":
    model.float()
else:
    clip.model.convert_weights(model)  # Default to float16, but convert back to float32 when needed

# Define loss functions
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

# Optimizer with learning rate scheduling
optimizer = optim.Adam(model.parameters(), lr=1e-5, betas=(0.9, 0.98), eps=1e-6)
num_epochs = 5  # Increase number of epochs for longer training

# Training loop
for epoch in range(num_epochs):
    for batch in train_dataloader:
        optimizer.zero_grad()

        images, texts = batch
        images = images.to(device)
        texts = texts.to(device)

        logits_per_image, logits_per_text = model(images, texts)

        ground_truth = torch.arange(len(images), dtype=torch.long, device=device)

        # Compute the loss
        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        total_loss.backward()

        if device == "cpu":
            optimizer.step()
        else:
            convert_models_to_fp32(model)  # Ensure precision handling before optimizer step
            optimizer.step()
            clip.model.convert_weights(model)  # Convert back to float16

    # Save model checkpoint after each epoch
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss.item(),
    }, f"model_checkpoint/model_epoch_{epoch}.pt")
    print(f"Epoch {epoch} completed, model checkpoint saved.")

