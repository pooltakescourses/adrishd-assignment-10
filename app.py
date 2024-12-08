from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
import os
import pandas as pd
import tqdm
import cosine_compare
import torch.nn.functional as F
import numpy as np
import open_clip
from PIL import Image

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads/'
IMAGE_DIRECTORY = 'image_directory/coco_images_resized/'  # Add images here for searching
EMBEDDING_PATH = './image_embeddings.pickle'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
df = None
model = None
tokenizer = None
preprocess = None

# Ensure necessary directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_DIRECTORY, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

def initiate():
  global df, model, preprocess, tokenizer
  df = pd.read_pickle(EMBEDDING_PATH)
  model, _, preprocess = open_clip.create_model_and_transforms('ViT-B/32', pretrained='openai')
  tokenizer = open_clip.get_tokenizer('ViT-B-32')



@app.route('/search', methods=['POST'])
def search():
    query_type = request.form.get('query_type')
    text_query = request.form.get('text_query')
    hybrid_weight = request.form.get('hybrid_weight', type=float)
    uploaded_file = request.files.get('image_query')
    
    uploaded_image_path = None
    if uploaded_file:
        filename = secure_filename(uploaded_file.filename)
        uploaded_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_file.save(uploaded_image_path)

    results = perform_search(query_type, text_query, uploaded_image_path, hybrid_weight)

    # Return JSON response
    return jsonify({"results": results})

def perform_search(query_type, text_query, image_path, hybrid_weight):
    """
    Placeholder for image/text search logic. 
    Replace this function with actual implementation.

    Returns:
        List[dict]: A list of search results with image paths and similarity scores.
    """
    vector = None
    if query_type == "text_query":
      text = tokenizer(text_query)
      vector = F.normalize(model.encode_text(text))

    elif query_type == "image_query":
      image = preprocess(Image.open(image_path)).unsqueeze(0)
      vector = F.normalize(model.encode_image(image))

    elif query_type == "hybrid_query":
      text = tokenizer(text_query)
      text_vec = F.normalize(model.encode_text(text))
      image = preprocess(Image.open(image_path)).unsqueeze(0)
      image_vec = F.normalize(model.encode_image(image))
      vector = hybrid_weight * text_vec + (1 - hybrid_weight) * image_vec

    query_result = cosine_compare.compare(df, vector.detach())

    results = []

    for file, score in query_result.items():
      res = dict()
      res['image_path'] = IMAGE_DIRECTORY + file
      res['similarity'] = float(score)
      results.append(res)

    return results

with app.app_context():
  initiate()

if __name__ == '__main__':
    app.run(debug=True)
