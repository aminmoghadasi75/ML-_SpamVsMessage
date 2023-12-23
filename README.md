# SMS Spam Detection using Machine Learning

This repository contains code for detecting spam and legitimate messages in SMS using a Machine Learning approach.

## Overview

Spam messages are a prevalent issue in communication channels like SMS. This project aims to classify SMS messages as spam or legitimate using a machine learning model trained on a dataset containing labeled examples.

## Dataset

The dataset used for training and evaluation contains SMS messages labeled as spam or ham (legitimate). The dataset is provided as a text file and is loaded using Pandas for preprocessing and model training.

## Setup and Dependencies

### Libraries Used

- Pandas
- NumPy
- Seaborn
- Matplotlib
- Scikit-learn

### Installation

To run the code, ensure you have the required libraries installed. You can install them via pip:

- pip install pandas numpy seaborn matplotlib scikit-learn


## Usage

1. **Data Preparation**: The dataset is loaded using Pandas and preprocessed to create features and labels for training the model.
2. **Feature Extraction**: CountVectorizer from Scikit-learn is used to convert text data into numerical vectors.
3. **Model Training**: A Bernoulli Naive Bayes classifier is utilized for training the model on the extracted features.
4. **Evaluation**: The trained model is evaluated on a test set, and accuracy scores are computed to measure its performance.

## File Structure

- `README.md`: Overview of the project and instructions.
- `sms_spam_detection.ipynb`: Jupyter Notebook containing the code for data loading, preprocessing, model training, and evaluation.
- `data/`: Directory containing the dataset used for training and testing.

## Running the Code

1. Clone the repository:
```bash
git clone https://github.com/aminmoghadasi75/ML-_SpamVsMessage.git
```

2. Open the Jupyter Notebook (`sms_spam_detection.ipynb`) in your preferred environment (Jupyter Notebook, Google Colab, etc.).
3. Run the notebook cells sequentially to load data, train the model, and evaluate its performance.

## Results

The model achieves an accuracy score of `X%` on the test set, indicating its effectiveness in distinguishing between spam and legitimate SMS messages.


