import os
import shutil
import random
from pathlib import Path


SOURCE_DATASET = Path(r"D:\PlantVillage")
PROJECT_DIR = Path.cwd()

TRAIN_DIR = PROJECT_DIR / "dataset" / "train"
VALID_DIR = PROJECT_DIR / "dataset" / "valid"



CLASSES = [
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",

    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato__Target_Spot",
    "Tomato__Tomato_mosaic_virus",

    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy"
]



def create_folders():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VALID_DIR.mkdir(parents=True, exist_ok=True)

    print("Folders created successfully!")


def create_class_folders():
    for class_name in CLASSES:
        (TRAIN_DIR / class_name).mkdir(parents=True, exist_ok=True)
        (VALID_DIR / class_name).mkdir(parents=True, exist_ok=True)

    print("Class folders created successfully!")


def split_dataset():

    for class_name in CLASSES:

        source_folder = SOURCE_DATASET / class_name

        images = list(source_folder.glob("*"))

        random.shuffle(images)

        split_index = int(len(images) * 0.8)

        train_images = images[:split_index]
        valid_images = images[split_index:]

        for image in train_images:
            shutil.copy(image, TRAIN_DIR / class_name / image.name)

        for image in valid_images:
            shutil.copy(image, VALID_DIR / class_name / image.name)

        print(f"{class_name} completed")


if __name__ == "__main__":
    create_folders()
    create_class_folders()
    split_dataset()


