import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, InceptionV3, Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, AveragePooling2D, GlobalAveragePooling2D
import numpy as np
import pandas as pd
import h5py

# Dictionary for model selection
MODELS = {
    "vgg16": VGG16,
    "vgg19": VGG19,
    "inception": InceptionV3,
    "xception": Xception,
    "resnet": ResNet50,
}

def create_model(model_name, mode, lr, decay, dropout_rate):
    # Adjust image size based on the model
    img_size = (299, 299, 3) if model_name in ["inception", "xception"] else (224, 224, 3)
    
    # Initialize the model
    model = Sequential()
    
    # Configure base model
    if model_name in MODELS:
        base_model = MODELS[model_name](weights="imagenet", include_top=False, input_shape=img_size)
        base_model.trainable = False  # Freeze the base model
        model.add(base_model)
    elif model_name == "simple":
        model.add(Conv2D(32, (5, 5), strides=(3, 3), activation='relu', input_shape=img_size))
        model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
    else:
        raise ValueError("Unsupported model type")
    
    # Add global pooling and dense layers
    model.add(GlobalAveragePooling2D())
    model.add(Dense(512, activation='relu'))
    if mode == "train":
        model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=lr, decay=decay),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# Function to load data from HDF5 files
def load_data(filename, train=True):
    with h5py.File(filename, 'r') as f:
        images = np.array(f["images"])
        labels = np.array(f["labels"])

    if train:
        # Shuffle the dataset for training data
        indices = np.arange(images.shape[0])
        np.random.shuffle(indices)
        images = images[indices]
        labels = labels[indices]

    return images, labels

# Function to preprocess and create a dataset
def create_dataset(images, labels, model_name, batch_size, training=False):
    img_size = (299, 299) if model_name in ["inception", "xception"] else (224, 224)
    
    def preprocess_fn(image, label):
        image = tf.image.resize(image, img_size)
        image = tf.keras.applications.vgg16.preprocess_input(image) if model_name.startswith("vgg") else image
        return image, label

    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    dataset = dataset.map(preprocess_fn, num_parallel_calls=tf.data.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(1024).repeat()
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset
