import cv2
import numpy as np
from scipy import ndimage
from PIL import Image, ImageEnhance
import argparse
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
from PIL import Image
import matplotlib.pyplot as plt
import csv

IMAGE_DIM = 224
ZOOM_LEVEL = 17
LAT_INCREMENT = 0.00250746 / IMAGE_DIM
LNG_INCREMENT = 0.00274658 / IMAGE_DIM

class ImageProcessor:
    def __init__(self, model, test_images):
        self.model = model
        self.test_images = test_images

    def preprocess_image(self, image_path):
        """Load and preprocess the image."""
        img = Image.open(image_path).resize((IMAGE_DIM, IMAGE_DIM))
        img_array = np.array(img)
        return img_array

    def predict_and_mask(self, image_array):
        """Predict the class and create a mask based on the model's prediction."""
        prediction = self.model.predict(image_array[np.newaxis, ...])
        mask = (prediction > 0.5).astype(np.uint8)  # Example thresholding
        return mask

    def enhance_image(self, image):
        """Enhance the contrast of the image."""
        enhancer = ImageEnhance.Contrast(image)
        enhanced_image = enhancer.enhance(2.0)
        return enhanced_image

    def process_images(self):
        """Main function to process and mask images."""
        results = {}
        for image_path in self.test_images:
            img_array = self.preprocess_image(image_path)
            mask = self.predict_and_mask(img_array)
            results[image_path] = mask
        return results

# Function for computing centroids and converting to latitude and longitude
def centroids_to_latlng(centroids, origin_lat, origin_lng):
    latlngs = []
    for centroid in centroids:
        lat = origin_lat + centroid[0] * LAT_INCREMENT
        lng = origin_lng + centroid[1] * LNG_INCREMENT
        latlngs.append((lat, lng))
    return latlngs

class ConnectedComponents:
    def __init__(self, binary_image):
        self.binary_image = binary_image

    def find_components(self):
        """Find connected components in the binary image."""
        structure = np.ones((3, 3), np.int)
        labeled, ncomponents = ndimage.label(self.binary_image, structure)
        return labeled, ncomponents

    def compute_centroids(self, labeled_image, ncomponents):
        """Compute the centroids of the connected components."""
        centroids = ndimage.center_of_mass(self.binary_image, labeled_image, range(1, ncomponents + 1))
        return centroids

# TEMPLATE FOR SHAPE CLASSIFICATION AND TRAINING



# def parse_arguments():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--save_dir', type=str, default='keras_model_simple')
#     parser.add_argument('--lr', type=float, default=2e-5)
#     parser.add_argument('--batch_size', type=int, default=24)
#     parser.add_argument('--dropout_rate', type=float, default=0.5)
#     parser.add_argument('--mode', type=str, default='train')
#     parser.add_argument('--eval_csv', type=str, default='../data/test_6_2019.csv')
#     parser.add_argument('--save_cc_img', action='store_true')
#     return vars(parser.parse_args())

# # Model creation
# def create_model(input_shape, mode, lr, dropout_rate):
#     conv_base = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
#     model = models.Sequential([
#         conv_base,
#         layers.GlobalAveragePooling2D(),
#         layers.Dense(512, activation='relu'),
#         layers.Dropout(dropout_rate) if mode == 'train' else layers.Layer(),
#         layers.Dense(1, activation='sigmoid')
#     ])
#     model.compile(loss='binary_crossentropy',
#                   optimizer=tf.keras.optimizers.Adam(lr=lr),
#                   metrics=['accuracy'])
#     return model

# # Image processing function
# def process_image(filename, img_size):
#     img = tf.io.read_file(filename)
#     img = tf.image.decode_jpeg(img, channels=3)
#     img = tf.image.resize(img, img_size[:2])
#     img = tf.keras.applications.vgg16.preprocess_input(img)
#     return img

# # Load and preprocess dataset
# def load_dataset(file_paths, labels, batch_size, img_size):
#     dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
#     dataset = dataset.map(lambda x, y: (process_image(x, img_size), y))
#     dataset = dataset.batch(batch_size)
#     return dataset

# Main function varies here based on use case. Adapt as needed