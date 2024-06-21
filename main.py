import os
import requests
import json
import cv2
import dlib
import numpy as np
import uuid
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# Initialize dlib's face detector and the facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("DAT\\shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("DAT\\dlib_face_recognition_resnet_model_v1.dat")

# Initialize QdrantApiClient with base URL
BASE_URL = "http://localhost:32771"
client = QdrantClient(url=BASE_URL)

# Function to create a collection in Qdrant
def create_collection(collection_name):
    try:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=128, distance=Distance.DOT),
        )
        print(f"Collection '{collection_name}' created successfully.")
    except Exception as e:
        print(f"Error creating collection '{collection_name}': {e}")

# Function to upload face encodings to Qdrant
def upload_to_qdrant(encoding, uuid, image_filename, collection_name):
    try:
        operation_info = client.upsert(
            collection_name=collection_name,
            wait=True,
            points=[PointStruct(id=str(uuid), vector=encoding.tolist(), payload={"image": image_filename})]
        )
        print(f"Uploaded {uuid} (from image '{image_filename}') to collection '{collection_name}' with status code {operation_info}")
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

# Example usage: Provide the folder path containing images and the collection name
folder_path = "combined"
collection_name = "Leaderguys"

# Create the collection in Qdrant
create_collection(collection_name)

# Get face encodings from images and upload them to the specified collection
uuid_mapping = get_face_encodings_from_folder(folder_path, collection_name)
print("Face UUID mapping:", uuid_mapping)
