import streamlit as st
import os
import pandas as pd
import numpy as np
import torch
import clip
from skimage import io
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import normalize


# Set device for running the model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/16", device=device, jit=False)
checkpoint = torch.load("model_checkpoint/model_epoch_4.pt")
model.load_state_dict(checkpoint['model_state_dict'])



# Function to load data from a TSV file
@st.cache(allow_output_mutation=True)
def load_data_from_tsv(tsv_file):
    df = pd.read_csv(tsv_file, sep='\t')  # Read the TSV file
    # Define valid image extensions
    valid_extensions = ('.jpg', '.jpeg', '.png')
    # Filter image paths to include only JPG, JPEG, and PNG files
    image_paths = [path for path in df['image_path'].tolist() if path.lower().endswith(valid_extensions)]
    # Keep captions aligned with filtered image paths
    captions = df['caption'].tolist()[:len(image_paths)]
    return image_paths, captions # Return lists of image paths and labels

# Synonyms and variants dictionary
variants = {
    'amchur': ['amchoor', 'amchur', 'dried mango powder'],
    # Add more variations here
}

# Function to normalize query
def normalize_query(query):
    query = query.lower()
    for canonical, synonyms in variants.items():
        if query in synonyms:
            return canonical
    return query

# BM25-like search on labels
def bm25_search(query, labels):
    vectorizer = CountVectorizer()
    tfidf_transformer = TfidfTransformer()

    X_counts = vectorizer.fit_transform(labels)
    X_tfidf = tfidf_transformer.fit_transform(X_counts)

    query_vec = vectorizer.transform([query])
    query_tfidf = tfidf_transformer.transform(query_vec)

    similarities = cosine_similarity(query_tfidf, X_tfidf).flatten()

    return similarities

# CLIP-based similarity search
def clip_search(path_to_encoded_images, query):
    # Load encoded image embeddings from the specified .npy file
    encoded_images = np.load(path_to_encoded_images, allow_pickle=True)

    text = clip.tokenize([query]).to(device)
    with torch.no_grad():
        emb = model.encode_text(text).cpu()

    sim_arr = cosine_similarity(emb, encoded_images)[0]

    return sim_arr

# Hybrid search combining BM25 and CLIP results
def hybrid_search(path_to_encoded_images, query, labels, weight_clip=0.5, weight_bm25=0.5):
    bm25_scores = bm25_search(query, labels)
    clip_scores = clip_search(path_to_encoded_images, query)

    # Normalize the scores
    bm25_scores = normalize(bm25_scores[:, np.newaxis], axis=0).ravel()
    clip_scores = normalize(clip_scores[:, np.newaxis], axis=0).ravel()

    hybrid_scores = weight_clip * clip_scores + weight_bm25 * bm25_scores
    sorted_indices = hybrid_scores.argsort()[::-1]

    return sorted_indices

# Load data and labels from TSV
tsv_file_path = 'data/train.tsv'  # Path to the TSV file
images_paths, labels = load_data_from_tsv(tsv_file_path)

# Ensure to handle the case where no images are loaded
if not images_paths:
    st.warning("No JPG images found in the provided TSV file.")

# Set the path to the encoded images
path_to_encoded_images = 'image_embeddings.npy'  # Path to the image embeddings .npy file

# Streamlit UI
st.title("Hybrid Image Search using CLIP and BM25")

# Text input for search query
query = st.text_input("Enter a search query) :", value="", key="query")

# Normalize the query on every input
normalized_query = normalize_query(query)

# Initialize session state for number of images, sliders, and search trigger
if "num_images" not in st.session_state:
    st.session_state.num_images = 10

if "weight_clip" not in st.session_state:
    st.session_state.weight_clip = 0.5

if "weight_bm25" not in st.session_state:
    st.session_state.weight_bm25 = 0.5

# Store slider values in session state
num_images = st.selectbox("How many top images to display?", [10, 20, 30, 40, 50], index=0, key="num_images_select")
st.session_state.num_images = num_images

weight_clip = st.slider("CLIP Embeddings Weight", 0.0, 1.0, st.session_state.weight_clip, 0.01, key="clip_weight")
st.session_state.weight_clip = weight_clip

weight_bm25 = st.slider("BM25 Labels Weight", 0.0, 1.0, st.session_state.weight_bm25, 0.01, key="bm25_weight")
st.session_state.weight_bm25 = weight_bm25

# Store the search button click state
if "search_clicked" not in st.session_state:
    st.session_state.search_clicked = False

# Button to trigger the search
if st.button("Search"):
    st.session_state.search_clicked = True

# Perform search only if the button is clicked and query is provided
if st.session_state.search_clicked and query:
    # Perform the hybrid search with the normalized query
    indices_array = hybrid_search(path_to_encoded_images, normalized_query, labels, 
                                  st.session_state.weight_clip, st.session_state.weight_bm25)
    
    # Load top images based on the sorted indices
    sorted_images_paths = [images_paths[i] for i in indices_array[:st.session_state.num_images]]
    sorted_imgs = [io.imread(path) for path in sorted_images_paths]

    # Display the top images
    st.write(f"Displaying top {st.session_state.num_images} images for the query: **{query}**")

    rows = (st.session_state.num_images + 4) // 5
    fig, ax = plt.subplots(nrows=rows, ncols=5, figsize=(10, 2 * rows))
    ax = ax.flatten()

    for i in range(st.session_state.num_images):
        ax[i].imshow(sorted_imgs[i])
        ax[i].axis('off')

    # Remove unused subplots
    for i in range(st.session_state.num_images, len(ax)):
        fig.delaxes(ax[i])

    st.pyplot(fig)

# Clear the search state if sliders or query change
else:
    st.write("Please enter a search query and click on the 'Search' button.")

