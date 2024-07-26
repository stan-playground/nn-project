import os

# Define the directory structure
directories = [
    "images",
    "models",
    "notebooks",
    "pages"
]

# Define the files to be created in each directory
files = {
    "images": ["logo.jpg", "metrics.jpg", "architecure.jpg"],
    "models": ["resnet.py", "resnet_weights.pt", "yolo_weights.pt"],
    "notebooks": ["resnet_train.ipynb", "yolo_train.ipynb"],
    "pages": ["image_classification.py", "object_detections.py"],
    "": ["app.py", "README.md"]  # Files in the root directory
}

# Create directories and files
for directory in directories:
    os.makedirs(directory, exist_ok=True)
    for file in files.get(directory, []):
        open(os.path.join(directory, file), 'a').close()

# Create files in the root directory
for file in files.get("", []):
    open(file, 'a').close()

print("Directory structure created successfully.")
