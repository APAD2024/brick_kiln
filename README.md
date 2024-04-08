# Advanced Kiln Detection System

## Overview

This repository is dedicated to the development and deployment of a cutting-edge kiln detection system, leveraging the power of machine learning and deep neural networks to analyze satellite imagery. With the use of advanced architectures like VGG16, VGG19, ResNet50, InceptionV3, and Xception, our system can accurately identify, evaluate, and track kiln structures across diverse geographical landscapes.

## Components

### Scripts

- **`eval_util.py`**: A utility script that defines a suite of functions and models essential for the evaluation process. It facilitates image preprocessing, model instantiation based on specified architectures, and the loading of trained models for further analysis.

- **`eval.py`**: The main evaluation driver script that integrates with `eval_util.py` to perform comprehensive model evaluations. It supports various command-line arguments for flexible evaluation configurations, including model selection, directory specifications, and performance metrics output.

- **`track_kiln.py`**: Specialized in tracking kiln locations within satellite images. It preprocesses images for optimal model consumption, applies detection algorithms, and translates detected kiln positions into accurate geographical coordinates.

### Features

- Multiple Deep Learning Architectures: Choose from a variety of pre-trained models to suit the specificity and complexity of your kiln detection tasks.
- Customizable Evaluation Metrics: Tailor your evaluation process with adjustable metrics to derive the most relevant performance insights.
- Geolocational Tracking: Convert image-based detections into precise geographical locations for comprehensive kiln tracking over time.

## Installation

Ensure you have Python 3.x installed on your system. Clone this repository and navigate into the project directory. Install all dependencies using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Model Evaluation

Perform an evaluation run using a specific model and dataset:

```bash
python eval.py --model <model_name> --save_dir ./models --test_filename ./data/test_set.h5
```

Supported models include: `vgg16`, `vgg19`, `resnet50`, `inceptionv3`, and `xception`.

### Kiln Tracking

Track kiln locations within an image:

```bash
python track_kiln.py --model <model_name> --save_dir ./models --image_path ./images/kiln.jpg
```

## Contributing

Contributions to the Advanced Kiln Detection System are welcome! Please refer to the CONTRIBUTING.md for guidelines on how to make a contribution.

## License

This project is released under the MIT License. Please see the [LICENSE](LICENSE.md) file for more details.

## Acknowledgments

- TensorFlow and Keras Teams for providing the frameworks and pre-trained models.
- Satellite imagery providers for enabling the application of machine learning techniques in geolocational analyses.
