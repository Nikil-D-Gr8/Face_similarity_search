# Face_similarity_search

This project demonstrates how to use Qdrant for storing and querying face encodings using Python.

## Prerequisites

- Docker installed and running on your system
- Python 3.x installed
- Required Python packages:
  - `opencv-python`
  - `dlib`
  - `numpy`
  - `requests`
  - `qdrant-client`
- Pre-trained models:
  - `shape_predictor_68_face_landmarks.dat`
  - `dlib_face_recognition_resnet_model_v1.dat`

## Quickstart

### 1. Run Qdrant using Docker

First, download the latest Qdrant image from Dockerhub:

```sh
docker pull qdrant/qdrant
```

Then, run the service:

```sh
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

### 2. Store Face Encodings

1. Place your images in a folder, e.g., `combined`.

2. Modify the `folder_path` and `collection_name` variables in the script as needed.

3. Run the script to create the collection and upload face encodings:

```sh
python main.py
```

### 3. Query Similar Faces

1. Modify the `image_path` and `collection_name` variables in the script as needed.

3. Run the script to query the Qdrant collection for similar faces:

```sh
python query.py
```

## Notes

- Ensure that the paths to the pre-trained models (`shape_predictor_68_face_landmarks.dat` and `dlib_face_recognition_resnet_model_v1.dat`) are correct in both scripts.
- Ensure that the Qdrant service is running and accessible at the specified `BASE_URL`.Sometimes the port number might change so be mindful of it.
- Adjust the collection name in both scripts as needed.
- Use the appropriate Docker ports and paths based on your local setup.

For any issues or further assistance, please refer to the [Qdrant documentation](https://qdrant.tech/documentation/) or open an issue in this repository.
