import tensorflow as tf
from tensorflow.keras.applications import VGG16, VGG19, ResNet50, InceptionV3, Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import numpy as np
import pandas as pd
import os
import time
import argparse
from tqdm import tqdm
from eval_util import *


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, default="vgg16")
    ap.add_argument("--save_dir", type=str, default="keras_model")
    ap.add_argument("--decay", type=float, default=1e-7)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--num_epochs", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--dropout_rate", type=float, default=0.6)
    ap.add_argument("--max_steps", type=int, default=1000)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--mode", type=str, default="train")
    ap.add_argument("--train_filename", type=str, default="")
    ap.add_argument("--test_filename", type=str, default="")
    ap.add_argument("--results_csv", type=str, default="")
    return vars(ap.parse_args())

def create_model(model_name, input_shape=(224, 224, 3)):
    MODELS = {"vgg16": VGG16, "vgg19": VGG19, "inception": InceptionV3,
              "xception": Xception, "resnet": ResNet50}

    if model_name not in MODELS:
        raise ValueError(f"Model {model_name} is not supported.")

    base_model = MODELS[model_name](weights="imagenet", include_top=False, input_shape=input_shape)
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1, activation='sigmoid')
    ])

    return model

def compile_model(model, learning_rate, decay):
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate, decay=decay),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

def load_data(filename):
    with np.load(filename) as data:
        return data['images'], data['labels']

def prepare_dataset(images, labels, batch_size, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices((images, labels))
    if augment:
        dataset = dataset.map(lambda x, y: (data_augmentation(x), y), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

def train_model(model, train_dataset, val_dataset, epochs, save_dir):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath=save_dir, save_best_only=True),
        tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')
    ]
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=epochs, callbacks=callbacks)
    return history

def evaluate_model(model, test_dataset):
    results = model.evaluate(test_dataset)
    print(f"Test Loss, Test Accuracy: {results}")

def main(args):
    model = create_model(args['model'])
    compile_model(model, args['lr'], args['decay'])

    if args['mode'] == 'train':
        train_images, train_labels = load_data(args['train_filename'])
        val_images, val_labels = load_data(args['test_filename'])
        
        train_dataset = prepare_dataset(train_images, train_labels, args['batch_size'], augment=True)
        val_dataset = prepare_dataset(val_images, val_labels, args['batch_size'])
        
        train_model(model, train_dataset, val_dataset, args['num_epochs'], args['save_dir'])

    elif args['mode'] in ['test', 'eval']:
        test_images, test_labels = load_data(args['test_filename'])
        test_dataset = prepare_dataset(test_images, test_labels, args['batch_size'])
        evaluate_model(model, test_dataset)

if __name__ == '__main__':
    args = parse_args()
    main(args)
