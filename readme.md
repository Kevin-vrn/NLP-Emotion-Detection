# Emotion Detection from Text Using NLP and Machine Learning

## 1. Project Overview

This project focuses on detecting emotions expressed in textual data using Natural Language Processing (NLP) and machine learning techniques.

The system processes text, converts it into numerical features using TF-IDF, and classifies the text into one of six emotion categories:

- Sadness
- Joy
- Love
- Anger
- Fear
- Surprise

Three machine learning algorithms were implemented and compared:

- Multinomial Naive Bayes
- Logistic Regression
- Linear Support Vector Machine (SVM)

Based on the validation results, Linear SVM was selected as the final model.

---

## 2. Objectives

The main objectives of this project are:

- To develop an NLP-based system for emotion detection from text.
- To preprocess textual data for machine learning.
- To extract useful numerical features using TF-IDF.
- To implement and compare different machine learning classifiers.
- To evaluate the performance of the models using standard classification metrics.
- To analyze the errors made by the final model.
- To identify possible improvements for future development.

---

## 3. Dataset

The dataset contains text samples along with their corresponding emotion labels.

The dataset is divided into three parts:

| Dataset | Records | Purpose |
|---|---:|---|
| Training | 16,000 | Model training |
| Validation | 2,000 | Model comparison and selection |
| Test | 2,000 | Final evaluation |

The six emotion classes used in the project are:

| Label | Emotion |
|---:|---|
| 0 | Sadness |
| 1 | Joy |
| 2 | Love |
| 3 | Anger |
| 4 | Fear |
| 5 | Surprise |

---

## 4. Methodology

The project follows these main steps:

1. Data collection and preparation
2. Text preprocessing
3. TF-IDF feature extraction
4. Model training
5. Model comparison using validation data
6. Selection of the best-performing model
7. Final testing on unseen test data
8. Error analysis

### Text Preprocessing

The following preprocessing operations were applied:

- Conversion of text to lowercase
- Removal of non-alphabetic characters
- Tokenization
- Removal of English stop words
- Creation of cleaned text

### Feature Extraction

TF-IDF (Term Frequency-Inverse Document Frequency) was used to convert the cleaned text into numerical features.

The implementation considers both unigrams and bigrams to capture individual words and short word combinations.

---

## 5. Machine Learning Models

Three machine learning algorithms were implemented:

### Multinomial Naive Bayes

A probabilistic classification algorithm used as a baseline model for text classification.

### Logistic Regression

A linear classification algorithm used to classify the TF-IDF feature vectors into the six emotion categories.

### Linear SVM

A linear Support Vector Machine used for text classification. It achieved the best overall validation performance and was selected as the final model.

---

## 6. Model Comparison

The models were compared using the validation dataset.

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
|---|---:|---:|---:|---:|
| Multinomial Naive Bayes | 75.80% | 87.99% | 54.55% | 58.01% |
| Logistic Regression | 89.20% | 89.76% | 81.55% | 84.90% |
| Linear SVM | 90.60% | 88.27% | 86.41% | 87.30% |

Linear SVM achieved the highest validation accuracy and macro F1-score and was therefore selected for final testing.

---

## 7. Final Results

The selected Linear SVM model was evaluated on 2,000 unseen test samples.

The model achieved an overall accuracy of:

**90.10%**

### Classification Performance

| Emotion | Precision | Recall | F1-Score |
|---|---:|---:|---:|
| Sadness | 94% | 94% | 94% |
| Joy | 92% | 94% | 93% |
| Love | 79% | 80% | 80% |
| Anger | 89% | 90% | 89% |
| Fear | 89% | 83% | 86% |
| Surprise | 74% | 65% | 69% |

The model performed particularly well for Sadness and Joy, while Surprise was the most difficult emotion to classify.

---

## 8. Error Analysis

The final model produced 198 incorrect predictions out of 2,000 test samples.

The most common confusion was between Joy and Love:

- Joy → Love: 31 samples
- Love → Joy: 29 samples

Other noticeable confusions included:

- Sadness → Anger
- Fear → Sadness
- Anger → Sadness
- Surprise → Fear
- Fear → Surprise

These errors indicate that the model can have difficulty distinguishing between emotions with similar or overlapping textual expressions.

---

## 9. Limitations

The project has some limitations:

- Some emotions have similar textual expressions and can be difficult to distinguish.
- The Surprise category achieved lower performance than several other emotion classes.
- TF-IDF has limited ability to understand deeper contextual meaning.
- The model may struggle with text where emotional meaning depends heavily on context.
- The system supports only six predefined emotion categories.
- Model performance depends on the quality and characteristics of the dataset.

---

## 10. Future Scope

The project can be improved in the future by:

- Using larger and more diverse datasets.
- Exploring advanced NLP techniques.
- Applying deep learning and transformer-based models.
- Supporting multiple languages.
- Developing a real-time emotion detection application.
- Creating a web or mobile application.
- Expanding the number of emotion categories.
- Improving the handling of ambiguous and difficult text samples.

---

## 11. Project Structure

```text
NLP-Emotion-Detection/
│
├── README.md
├── requirements.txt
├── pre.py
├── train_and_evaluate.py
├── predict.py
├── error_analysis.py
│
├── dataset/
│   ├── train.csv
│   ├── train_processed.csv
│   ├── validation.csv
│   └── test.csv
│
├── models/
│   ├── svm_model.pkl
│   └── tfidf_vectorizer.pkl
│
├── results/
│   ├── model_comparison.csv
│   ├── classification_report.txt
│   ├── final_confusion_matrix.png
│   ├── test_predictions.csv
│   ├── incorrect_test_predictions.csv
│   ├── test_confusion_pairs.csv
│   ├── top_confusion_pairs.png
│   └── model_accuracy_comparison.png
│
├── screenshots/
└── report/
    └── NLP_Emotion_Detection_Report.pdf