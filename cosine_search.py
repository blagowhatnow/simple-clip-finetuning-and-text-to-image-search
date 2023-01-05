import skimage
from skimage import io
import os
from pathlib import Path
import numpy as np
import torch
import clip
from PIL import Image

device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT/B-32",device=device,jit=False) #Must set jit=False for training
checkpoint = torch.load("model_checkpoint/model_10.pt")


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
    return images

ROOT_PATH = "./"
train_data = os.path.join(ROOT_PATH, "../ImageBank")

images=load_data(train_data)


from sklearn.metrics.pairwise import cosine_similarity


def sorted_by_similarities(text, path_to_encoded):
  #Load input encodings
  with open(str(path_to_encoded), 'rb') as f:
       encoded_images = np.load(f, allow_pickle=True)
  #encode text
  text = clip.tokenize(["yellow print sleeveless men tshirt"]).to(device)
  with torch.no_grad():
      emb = model.encode_text(text).cpu()
  #compute similarity array
  sim_arr = cosine_similarity(emb, encoded_images)[0]
  #sort array, get indices and sort encoded array by indices in descending order
  sorted_inds=sim_arr.argsort()[::-1]
  return sorted_inds


#Make plots

indices_array=sorted_by_similarities(text,path)

sorted_imgs=[images[i] for i in indices_array]

#Show first 25 images

import matplotlib.pyplot as plt

plt.figure(figsize=(10,10)) # specifying the overall grid size

for i in range(25):
    plt.subplot(5,5,i+1)    # the number of images in the grid is 5*5 (25)
    plt.imshow(sorted_imgs[i])

plt.savefig("results.png")
