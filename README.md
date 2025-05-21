# 📧 Spam Mail Detection 

A Machine Learning-based project to detect **spam emails** using text classification with **TF-IDF vectorization** and a **Random Forest Classifier**. This project reads email text data, preprocesses it, trains a model, and predicts whether an input email is spam or not.

---

## 🔍 Project Overview

- ✅ **Preprocessing:** Clean and encode email data
- 📈 **Feature Engineering:** Use `TfidfVectorizer` to convert email text to numeric vectors
- 🌲 **Model:** Train a `RandomForestClassifier` for classification
- 🧪 **Evaluation:** Measure training and test accuracy
- 📬 **Prediction:** Classify new/unseen messages as **Spam** or **Ham**

---


## 📂 Dataset

The dataset (`mail_data.csv`) contains:
- **Category**: Label indicating whether the email is spam (0) or ham (1)
- **Message**: The actual email content

## 📈 Model Workflow

1. **Label Encoding**  
   - 'spam' ➝ 0  
   - 'ham' ➝ 1

2. **Splitting Data**  
   - 80% for training  
   - 20% for testing

3. **TF-IDF Vectorization**  
   - Text is converted to numerical format using `TfidfVectorizer`

4. **Model Training**  
   - Trained a `RandomForestClassifier` for binary classification

---

