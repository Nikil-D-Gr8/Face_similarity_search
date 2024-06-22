import os
import cv2
import dlib
import numpy as np
import uuid
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import gradio as gr
from PIL import Image
import json
from password import URL ,APIKEY

# Initialize dlib's face detector and the facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("DAT/shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("DAT/dlib_face_recognition_resnet_model_v1.dat")

# Initialize QdrantApiClient with base URL
client = QdrantClient(
    url=URL, 
    api_key=APIKEY,
)

CONFIG_FILE = 'config.json'

# Function to load configuration from JSON file
def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as file:
            return json.load(file)
    else:
        return {}

# Function to save configuration to JSON file
def save_config(config):
    with open(CONFIG_FILE, 'w') as file:
        json.dump(config, file, indent=4)

# Function to create a collection in Qdrant
def create_collection(collection_name):
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=128, distance=Distance.DOT),
        )
        return f"Collection '{collection_name}' created successfully."
    except Exception as e:
        return f"Error creating collection '{collection_name}': {e}"

# Function to upload face encodings to Qdrant
def upload_to_qdrant(encoding, uuid, image_filename, collection_name):
    try:
        operation_info = client.upsert(
            collection_name=collection_name,
            wait=True,
            points=[PointStruct(id=str(uuid), vector=encoding.tolist(), payload={"image": image_filename})]
        )
    except Exception as e:
        print(f"Error uploading {uuid} (from image '{image_filename}') to collection '{collection_name}': {e}")

# Function to get face encodings for all images in a folder
def get_face_encodings_from_folder(folder_path, collection_name):
    uuid_mapping = {}  # To store UUIDs mapped to image filenames
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(folder_path, filename)
            encodings = get_face_encodings(image_path)
            
            if encodings:
                # Generate UUID for each face in the image
                face_uuids = [uuid.uuid4() for _ in range(len(encodings))]
                
                # Upload face encodings to Qdrant with respective UUIDs and image filenames
                for encoding, face_uuid in zip(encodings, face_uuids):
                    upload_to_qdrant(encoding, face_uuid, filename, collection_name)
                    uuid_mapping[str(face_uuid)] = filename  # Map UUID to image filename
    
    return uuid_mapping

# Function to get face encodings from a single image
def get_face_encodings(image_path):
    # Load the image using OpenCV
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Detect faces
    faces = detector(img_rgb, 1)

    # Initialize an empty list to store face encodings
    face_encodings = []

    for face in faces:
        # Get the landmarks/parts for the face
        shape = predictor(img_rgb, face)

        # Get the face encoding
        face_encoding = np.array(face_rec_model.compute_face_descriptor(img_rgb, shape, 1))
        face_encodings.append(face_encoding)

    return face_encodings

# Function to query Qdrant collection for similar vectors
def query_qdrant_collection(collection_name, query_vector, limit=100):
    try:
        search_result = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )

        results = [{'image': result.payload['image']} for result in search_result]
        return results
    
    except Exception as e:
        return f"Error querying collection '{collection_name}': {e}"

def scan_and_upload(folder_path):
    collection_name = datetime.now().strftime("%Y%m%d%H%M%S")
    create_collection(collection_name)
    uuid_mapping = get_face_encodings_from_folder(folder_path, collection_name)
    
    # Update and save config
    config = load_config()
    config[folder_path] = collection_name
    save_config(config)
    
    return f"Scanned and uploaded images from {folder_path} to collection '{collection_name}'.", uuid_mapping, collection_name

def query_and_display(image_path, folder_path):
    # Load config and get collection name
    config = load_config()
    collection_name = config.get(folder_path)
    
    if not collection_name:
        return f"No collection found for folder '{folder_path}'.", []
    
    encodings = get_face_encodings(image_path)
    
    if not encodings:
        return "No faces found in the uploaded image.", []
    
    query_vector = encodings[0].tolist()  # Use the first face encoding as the query vector
    results = query_qdrant_collection(collection_name, query_vector)
    return results

# Gradio Interface
# Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("### Face Recognition with Qdrant")
    
    with gr.Tab("Scan and Upload"):
        folder_input = gr.Textbox(label="Folder Path", placeholder="Enter the folder path containing images")
        upload_button = gr.Button("Scan and Upload")
        upload_output = gr.Textbox(label="Output")
        collection_name_display = gr.Textbox(label="Collection Name", interactive=False)
        upload_button.click(scan_and_upload, inputs=folder_input, outputs=[upload_output, collection_name_display, collection_name_display])
    
    with gr.Tab("Query"):
        folder_name_input = gr.Textbox(label="Folder Path", placeholder="Enter the folder path")
        image_input = gr.Image(type="filepath", label="Upload Image for Query")
        query_button = gr.Button("Query")
        query_results = gr.Gallery(label="Matching Images", columns=3, object_fit="contain")
        
        def query_and_display_with_gallery(image_path, folder_path):
            result_names = query_and_display(image_path, folder_path)
            if isinstance(result_names, list):
                result_images = [Image.open(os.path.join(folder_path, result['image'])) for result in result_names]
                return result_images
            else:
                return []
        
        query_button.click(query_and_display_with_gallery, inputs=[image_input, folder_name_input], outputs=query_results)

demo.launch()
