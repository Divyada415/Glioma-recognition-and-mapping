import os
import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))


BASE_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "raw",
    "Epic and CSCR hospital Dataset",
    "Epic and CSCR hospital Dataset"
)

IMG_SIZE = 224

def load_data(split):
    images = []
    labels = []

    split_path = os.path.join(BASE_PATH, split)

    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Path not found: {split_path}")

    print(f"📂 Loading data from: {split_path}")

    for class_name in os.listdir(split_path):
        class_path = os.path.join(split_path, class_name)

        if not os.path.isdir(class_path):
            continue

        label = 1 if class_name.lower() == "glioma" else 0

        for file in os.listdir(class_path):
            if file.lower().endswith(('.jpg', '.png', '.jpeg', '.bmp', '.tif')):
                img_path = os.path.join(class_path, file)

                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0

                images.append(img)
                labels.append(label)

    X = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(labels)

    return X, y


print("Processing TRAIN data...")
X_train, y_train = load_data(r"C:\Users\Divya DA\OneDrive\Desktop\Transpert_ai_glioma\data\raw\Epic and CSCR hospital Dataset\Train")

print("Processing TEST data...")
X_test, y_test = load_data(r"C:\Users\Divya DA\OneDrive\Desktop\Transpert_ai_glioma\data\raw\Epic and CSCR hospital Dataset\Test")
"""
Stage 1 & 2: Data Acquisition and Preprocessing
- MRI collection
- Normalization
- Resampling
"""

import os
import cv2
import numpy as np

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(
    PROJECT_DIR, "data", "raw",
    "Epic and CSCR hospital Dataset",
    "Epic and CSCR hospital Dataset"
)

IMG_SIZE = 224

def load_data(split):
    images, labels = [], []
    split_path = os.path.join(BASE_PATH, split)

    for cls in os.listdir(split_path):
        cls_path = os.path.join(split_path, cls)
        label = 1 if cls.lower() == "glioma" else 0

        for f in os.listdir(cls_path):
            img = cv2.imread(os.path.join(cls_path, f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            img = img / 255.0
            images.append(img)
            labels.append(label)

    X = np.array(images).reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    y = np.array(labels)
    return X, y

X_train, y_train = load_data(r"C:\Users\Divya DA\OneDrive\Desktop\Transpert_ai_glioma\data\raw\Epic and CSCR hospital Dataset\Train")
X_test, y_test = load_data(r"C:\Users\Divya DA\OneDrive\Desktop\Transpert_ai_glioma\data\raw\Epic and CSCR hospital Dataset\Test")

os.makedirs(os.path.join(PROJECT_DIR, "data", "processed"), exist_ok=True)
np.save("data/processed/X_train.npy", X_train)
np.save("data/processed/y_train.npy", y_train)
np.save("data/processed/X_test.npy", X_test)
np.save("data/processed/y_test.npy", y_test)

print("✅ Preprocessing completed")

os.makedirs(os.path.join(PROJECT_DIR, "data", "processed"), exist_ok=True)

np.save(os.path.join(PROJECT_DIR, "data", "processed", "X_train.npy"), X_train)
np.save(os.path.join(PROJECT_DIR, "data", "processed", "y_train.npy"), y_train)
np.save(os.path.join(PROJECT_DIR, "data", "processed", "X_test.npy"), X_test)
np.save(os.path.join(PROJECT_DIR, "data", "processed", "y_test.npy"), y_test)

print("✅ Preprocessing completed successfully!")
print("Train shape:", X_train.shape, y_train.shape)
print("Test shape:", X_test.shape, y_test.shape)
