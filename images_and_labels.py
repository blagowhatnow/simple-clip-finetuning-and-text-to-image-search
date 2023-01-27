import os
from skimage import io


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
    return images, labels

ROOT_PATH = "./"
train_data_directory = os.path.join(ROOT_PATH, "color")

images,labels=load_data(train_data_directory)

#Delete quotations

labels=[i.replace("'","") for i in labels]

count = 0

for i in images:
    io.imsave('data/imagespng/'+str(count)+'.png',i)
    count+=1

count =0

for i in labels:
   t_file=open('data/labelspng/'+str(count)+'.txt','w')
   t_file.write(i)
   t_file.close()
   count+=1

