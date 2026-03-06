# Hate-speech-detection

## Project Overview

This project focuses on detecting hate speech in text using Machine Learning techniques. The system classifies text as **hate speech** or **non-hate speech** based on linguistic patterns and features extracted from the dataset.

## Workflow

### 1. Preprocessing

In this step, the raw text data is cleaned and prepared for analysis. The preprocessing includes:

* Converting text to lowercase
* Removing punctuation and special characters
* Removing stopwords
* Tokenization
* Text cleaning

### 2. Feature Extraction

After preprocessing, the text is converted into numerical features so that machine learning models can process the data.

Feature extraction method used:

* **TF-IDF (Term Frequency – Inverse Document Frequency)**

### 3. Model

A machine learning model is trained using the extracted features to classify the text.

Example models that can be used:

* Logistic Regression
* Naive Bayes
* Support Vector Machine (SVM)

The model learns patterns in the dataset and predicts whether a sentence contains hate speech.

## Technologies Used

* Python
* Machine Learning
* Natural Language Processing (NLP)
* Scikit-learn
* Pandas
* NumPy

## Project Structure

preprocessing.py – Text cleaning and preprocessing
feature_extraction.py – Converting text into numerical features
model.py – Training and prediction using machine learning models

## Conclusion

This project demonstrates how machine learning and natural language processing techniques can be used to automatically detect hate speech in textual data.
