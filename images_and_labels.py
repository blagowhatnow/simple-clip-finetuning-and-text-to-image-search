import os
from PIL import Image  # Use PIL for image saving
from concurrent.futures import ThreadPoolExecutor, as_completed
from skimage import io
import numpy as np

def create_directories():
    os.makedirs('data/imagespng', exist_ok=True)
    os.makedirs('data/labelspng', exist_ok=True)


def load_images_from_directory(label_directory, valid_image_extensions):
    """Helper function to load images and labels from a single directory."""
    file_names = [os.path.join(label_directory, f) 
                  for f in sorted(os.listdir(label_directory))
                  if any(f.lower().endswith(ext) for ext in valid_image_extensions)]
    
    images = [io.imread(f) for f in file_names]  # Load images as NumPy arrays
    labels = [os.path.basename(label_directory) for _ in file_names]  # Use folder name as label
    return images, labels

def load_data(data_directory):
    valid_image_extensions = ['.jpg', '.jpeg', '.png']  # Specify allowed extensions
    directories = [os.path.join(data_directory, d) 
                   for d in os.listdir(data_directory) 
                   if os.path.isdir(os.path.join(data_directory, d))]

    images = []
    labels = []

    # Use ThreadPoolExecutor to load images from multiple directories in parallel
    with ThreadPoolExecutor() as executor:
        # Submit tasks for each directory
        future_to_directory = {executor.submit(load_images_from_directory, d, valid_image_extensions): d for d in directories}
        
        for future in as_completed(future_to_directory):
            try:
                imgs, lbls = future.result()  # Get images and labels from each directory
                images.extend(imgs)  # Add to the main images list
                labels.extend(lbls)  # Add to the main labels list
            except Exception as exc:
                print(f"Error processing directory {future_to_directory[future]}: {exc}")

    return images, labels


def save_data(images, labels):
    # Save images and labels to their respective directories
    count = 0
    for img in images:
        # Convert NumPy array to PIL Image
        img_pil = Image.fromarray(np.uint8(img))
        img_pil.save(f'data/imagespng/{count}.png')  # Save the image using PIL's save method
        count += 1

    count = 0
    for label in labels:
        with open(f'data/labelspng/{count}.txt', 'w') as label_file:
            label_file.write(label)
        count += 1


# Main execution
if __name__ == "__main__":
    # Create necessary directories
    create_directories()

    ROOT_PATH = "./"
    train_data_directory = os.path.join(ROOT_PATH, "enter path here")

    # Load images and labels
    images, labels = load_data(train_data_directory)

    # Clean up any quotations in the labels
    labels = [label.replace("'", "") for label in labels]

    # Save the data into 'data/imagespng' and 'data/labelspng'
    save_data(images, labels)

