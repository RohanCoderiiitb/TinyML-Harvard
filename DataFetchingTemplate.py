import os
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

# Step 1: Set up Kaggle API
os.environ['KAGGLE_CONFIG_DIR'] = '/content'  # or any path where kaggle.json is uploaded

# Step 2: Authenticate
api = KaggleApi()
api.authenticate()

# Step 3: Download dataset
# Example: PlantVillage dataset (change to any image dataset you want)
dataset = 'spitfiregg/plantvillage-dataset'  # Get this from Kaggle URL
api.dataset_download_files(dataset, path='/content', unzip=True)

# Step 4: Check structure and prepare folders
import shutil

# Optional: Move into separate 'train' and 'validation' folders
# Make sure this part is customized to how the dataset is structured
base_dir = '/content/plantvillage'
train_dir = '/tmp/train'
val_dir = '/tmp/validation'

# Example split (YOU NEED TO IMPLEMENT LOGIC HERE IF NOT PROVIDED)
# Here's a rough template:
from sklearn.model_selection import train_test_split
import glob

classes = os.listdir(base_dir)
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

for cls in classes:
    img_paths = glob.glob(os.path.join(base_dir, cls, '*.jpg'))
    train_imgs, val_imgs = train_test_split(img_paths, test_size=0.2, random_state=42)

    os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
    os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

    for path in train_imgs:
        shutil.copy(path, os.path.join(train_dir, cls))

    for path in val_imgs:
        shutil.copy(path, os.path.join(val_dir, cls))

print("Data organized successfully.")
