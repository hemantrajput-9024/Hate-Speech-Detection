# Hate Speech Detection using NLP and Machine Learning

## 📌 Project Overview

This project is a **Hate Speech Detection system** built using Natural Language Processing (NLP) and Machine Learning.

The main purpose of this project is to classify text into different categories based on the label provided in the dataset. The project applies several NLP preprocessing techniques to convert raw text into a format that can be understood by a machine learning model.

The final model uses **TF-IDF Vectorization** and **Multinomial Naive Bayes** for text classification.

---

## 📊 Dataset

The project uses a Hate Speech dataset containing **440,906 text records**.

### Dataset Columns

| Column | Description |
|---|---|
| `Content` | Original text content |
| `Label` | Target/class label |
| `Content_int` | Integer/token representation of the content |

The dataset contains text examples that are classified according to their corresponding labels.

---

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- NLTK
- Scikit-learn
- Joblib
- Jupyter Notebook / Google Colab

### Machine Learning

- TF-IDF Vectorizer
- Multinomial Naive Bayes
- Train-Test Split
- Classification Report
- Accuracy Score

---

## 🔄 Project Workflow

The complete workflow of the project is:

```text
Dataset
   ↓
Data Loading
   ↓
Data Understanding
   ↓
Label Analysis
   ↓
Data Reduction / Balancing
   ↓
Text Cleaning
   ↓
Tokenization
   ↓
Stopword Removal
   ↓
Lemmatization
   ↓
Processed Text
   ↓
Train-Test Split
   ↓
TF-IDF Vectorization
   ↓
Multinomial Naive Bayes
   ↓
Prediction
   ↓
Model Evaluation
   ↓
Model Saving
```

---

## 🧹 Text Preprocessing

Text preprocessing is performed before training the machine learning model.

### 1. Text Cleaning

The text is converted to lowercase and unwanted non-alphabetical characters are removed.

```python
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text
```

### 2. Tokenization

The cleaned text is divided into individual words using NLTK's `word_tokenize`.

### 3. Stopword Removal

Common English stopwords are removed using NLTK.

Examples include:

```text
the
is
and
a
an
to
of
```

### 4. Lemmatization

Words are converted into their base form using `WordNetLemmatizer`.

This helps reduce different forms of the same word into a common representation.

---

## ✂️ Train-Test Split

The processed text is divided into training and testing datasets.

The project uses:

-
