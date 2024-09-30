import torch
import clip
from PIL import Image
import pandas as pd

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device)
checkpoint = torch.load("model_checkpoint/model_epoch_4.pt")
model.load_state_dict(checkpoint['model_state_dict'])

df=pd.read_csv("data/train.tsv", sep='\t')
labels=list(set(df['caption']))


image = preprocess(Image.open("data/imagespng/41.png")).unsqueeze(0).to(device)
text = clip.tokenize(labels).to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    
    logits_per_image, logits_per_text = model(image, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()

print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]

print(labels)

