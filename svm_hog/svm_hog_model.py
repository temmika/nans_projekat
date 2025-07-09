import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load

def load_images_and_labels(data_dir, img_size=(128, 128)):
    images = []
    labels = []
    for label in os.listdir(data_dir):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        for fname in os.listdir(label_path):
            fpath = os.path.join(label_path, fname)
            img = cv2.imread(fpath)
            if img is None:
                continue
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)
    return images, labels

def extract_hog_features(images):
    features = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_feat = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
        features.append(hog_feat)
    return np.array(features)

def train_svm_hog(train_dir):
    images, labels = load_images_and_labels(train_dir)
    features = extract_hog_features(images)
    le = LabelEncoder()
    y = le.fit_transform(labels)
    svm = LinearSVC(max_iter=10000)
    svm.fit(features, y)
    # Sačuvaj model i label encoder
    dump(svm, "svm_hog_model.joblib")
    dump(le, "label_encoder.joblib")
    print("Model treniran i sačuvan.")

def predict_image(img_path):
    svm = load("svm_hog_model.joblib")
    le = load("label_encoder.joblib")
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hog_feat = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), feature_vector=True)
    pred = svm.predict([hog_feat])
    return le.inverse_transform(pred)[0]

# Primer pokretanja treninga:
# train_svm_hog(r'data/data_processing/train')