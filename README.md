# Image Caption Generator using Deep Learning 🖼️


## Demo

Link: [Image Caption Generator Web App](#)

## Overview
This project is a deep learning-based image caption generator. It leverages pre-trained models like VGG16 for image feature extraction and LSTM (Long Short-Term Memory) networks for caption generation. Using the Flickr8k dataset, which includes 8,091 images and 40,455 captions, the model generates meaningful textual descriptions for any given image.

## Motivation
The inspiration behind this project was to explore a challenging aspect of artificial intelligence—image captioning. This task involves generating coherent and accurate captions for a variety of images, requiring a combination of computer vision and natural language processing. The idea was to utilize pre-trained models and deep learning frameworks to create an intuitive and effective solution for caption generation.

## Technical Aspect
The project is divided into several parts:

1. **Data Preparation**: Cleaning and preprocessing of the Flickr8k dataset, including tokenization, padding, and feature extraction using VGG16.
2. **Model Training**: Training a deep learning model that combines image features with LSTM for sequence prediction.
3. **Graphical User Interface**: Building a simple GUI using Tkinter to allow users to upload images and view generated captions.
4. **Model Evaluation**: Evaluating the model using BLEU scores to measure accuracy and consistency.
5. **Data Visualization**: Utilizing pandas, seaborn, and matplotlib for analyzing and visualizing the dataset.

## Installation
The project is developed in Python 3.7+. To install the required packages and libraries, follow these steps:

1. Clone the repository.
2. Navigate to the project directory.
3. Run the following command:

```bash
pip install -r requirements.txt
```

## Run
To execute the application locally, follow these instructions:

1. **Data Preprocessing**: Ensure the dataset is cleaned and preprocessed using the provided code.
2. **Model Training**: Train the model using the preprocessed dataset and save the trained weights.
3. **GUI Execution**: Run the GUI application using the following command:

```bash
python app/main.py
```

## Directory Tree
```
├── app
│   ├── __init__.py
│   ├── main.py
│   ├── model
│   │   └── model_weights.h5
│   ├── static
│   └── templates
├── config
│   ├── __init__.py
├── processing
│   ├── __init__.py
├── data
│   └── Flickr8k
├── requirements.txt
├── README.md

```

## Model Evaluation
The model's performance is evaluated using the BLEU (Bilingual Evaluation Understudy) metric, which compares the generated captions with reference captions. BLEU scores range from 0.0 (worst) to 1.0 (best), measuring the accuracy of the generated descriptions.

## Performance Analysis
The performance of the image caption generator was analyzed using both qualitative and quantitative approaches:

- **Visual Inspection**: Checking the coherence and accuracy of generated captions.
- **Evaluation Metrics**: BLEU scores provide a quantitative measure of model performance.
- **Further Improvements**: Suggestions were identified for improving accuracy and model efficiency, including experimenting with different architectures, adjusting hyperparameters, and cleaning the dataset more effectively.

## Technologies Used
- **Python**: Programming Language
- **TensorFlow & Keras**: Deep Learning Frameworks
- **OpenCV**: Computer Vision Library
- **Numpy & Pandas**: Data Manipulation
- **Matplotlib & Seaborn**: Data Visualization
- **Tkinter**: Graphical User Interface
- **Flask**: Web Application Framework
- **Heroku**: Deployment Platform


## Credits
- **Flickr8k Dataset**: Image dataset used for training and evaluation. Available on [Kaggle](https://www.kaggle.com/shadabhussain/flickr8k).
- **Pre-trained VGG16 Model**: Used for image feature extraction.
- **Keras**: For building and training the deep learning model.
- **Tkinter & PIL**: For GUI development.

