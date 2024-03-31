# Twitter Hate Speech Analysis

This project focuses on analyzing hate speech on Twitter using Logistic Regression algorithm.

## Introduction

Social media platforms, like Twitter, have become integral parts of modern communication. However, the rise of hate speech and abusive language on these platforms poses significant challenges. Understanding and effectively addressing hate speech is crucial for maintaining a healthy online environment. Machine learning techniques offer potential solutions for identifying and combating hate speech.

## Objective

The primary objective of this project is to develop a model that can accurately classify tweets as either containing hate speech or not, based on their content and context.

## Dataset

The dataset used in this project consists of tweets collected from Twitter, labeled as hate speech, offensive speech or neutral speech. It includes features such as text content, user information, and metadata. Preprocessing steps involve removal of punctuation and capitalization, tokenizing, removal of stopwords, and stemming.

Source: https://www.kaggle.com/datasets/mrmorj/hate-speech-and-offensive-language-dataset

## Algorithm Used

**Logistic Regression**: In this project, Logistic Regression is employed to build a predictive model for identifying hate speech on Twitter. This would be a multiclass classification - the model would be classifying tweets as hate speech, offensive speech, and neutral speech. 

## Implementation

The project implementation comprises the following steps:

1. Data Preprocessing: Cleaning and preprocessing the tweet data, including text normalization.
2. Feature Extraction: Extracting relevant features from the text data, such as TF-IDF vectors or word embeddings.
3. Model Training: Splitting the dataset into training and testing sets, training a Logistic Regression model on the training data.
4. Model Evaluation: Evaluating the performance of the trained model using the accuracy metric.
5. Model Interpretation: Analyzing the coefficients of the logistic regression model to understand the importance of different features in predicting hate speech.

## Usage

To utilize this project, follow these steps:

1. Clone the repository to your local machine.
2. Install the necessary dependencies such as nltk, scikit-learn, seaborn, textstat, numpy, matplotlib, vaderSentiment.
3. Execute the Jupyter notebook or Python script to preprocess the data, train the model, and evaluate its performance.

## Conclusion

By leveraging Logistic Regression, this project aims to contribute to the identification and mitigation of hate speech on Twitter. The developed model can assist social media platforms and policymakers in implementing measures to combat online abuse and foster a more inclusive online environment.
