import cv2
import dlib
import numpy as np
from qdrant_client import QdrantClient
from password import URL , APIKEY

# Initialize dlib's face detector and the facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("DAT\\shape_predictor_68_face_landmarks.dat")
face_rec_model = dlib.face_recognition_model_v1("DAT\\dlib_face_recognition_resnet_model_v1.dat")

# Initialize QdrantApiClient with base URL
client = QdrantClient(url=URL,api_key=APIKEY)

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

        print(f"Top {limit} results:")
        for i, result in enumerate(search_result):
            print(f"Result {i+1}: {result.payload}")
    
    except Exception as e:
        print(f"Error querying collection '{collection_name}': {e}")

# Example usage: Provide the image path and the collection name
image_path = "modicheck.jpg"
collection_name = "Leaderguys"

# Get face encodings from the image
encodings = get_face_encodings(image_path)

# If there are no faces detected, exit
if not encodings:
    print(f"No faces found in {image_path}.")
else:
    # Query the Qdrant collection with the first face encoding
    query_vector = encodings[0].tolist()  # Use the first face encoding as the query vector
    query_qdrant_collection(collection_name, query_vector)
