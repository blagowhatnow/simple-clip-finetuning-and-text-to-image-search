import skimage
from skimage import io
import clip
import torch
import os
from PIL import Image

torch.cuda.empty_cache()

def load_data(data_directory):
    directories = [d for d in os.listdir(data_directory)
                   if os.path.isdir(os.path.join(data_directory, d))]
    labels = []
    images = []
    for d in directories:
        label_directory = os.path.join(data_directory, d)
        file_names = [os.path.join(label_directory, f)
                      for f in sorted(os.listdir(label_directory))
                      if f.endswith(".jpg")]
        for f in file_names:
            images.append(io.imread(f))
            labels.append(d)
    return images,labels

ROOT_PATH = "./"
train_data = os.path.join(ROOT_PATH, "../color")

images,labels=load_data(train_data)
labels=[i.replace("'","") for i in labels]



device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16",device=device,jit=False) #Must set jit=False for training
checkpoint = torch.load("model_checkpoint/model_10.pt")

preprocessed=torch.cat([preprocess(Image.fromarray(i)).unsqueeze(0).to(device) for i in images])

with torch.no_grad():
    encoded=model.encode_image(preprocessed).cpu().detach().numpy().squeeze()



import numpy as np

with open('embeddings.npy', 'wb') as f:
    np.save(f, encoded ,allow_pickle=True)


