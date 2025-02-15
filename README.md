# 📰 Fake News Detection

## 📌 Overview
This project aims to classify news articles as **Real** or **Fake** using **Natural Language Processing (NLP)** and **Machine Learning**. The model is trained on a dataset of real and fake news articles to detect misinformation effectively.

## 🚀 Features
- Preprocessing of text data
- TF-IDF vectorization
- Machine Learning classification model
- Streamlit web app for user-friendly predictions

## 📂 Dataset
The dataset consists of two CSV files:
- **True.csv** - Contains real news articles
- **Fake.csv** - Contains fake news articles

> **Note:** Due to GitHub file size limits, the dataset is not included in the repository. You can download it from [Kaggle](https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset) and place it in the project directory.

## 🏗️ Tech Stack
- Python
- Pandas, NumPy, Scikit-learn
- Natural Language Processing (NLP)
- Streamlit (for Web UI)
- Git & GitHub

## ⚡ Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/akhildanday/Fake-News-Detection.git
   cd Fake-News-Detection
   ```

2. **Create and activate a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset** and place `True.csv` and `Fake.csv` in the project directory.

## 🏃 Usage
### 1️⃣ Train the Model
Run the following script to preprocess data, train the model, and save it:
```bash
python train.py
```

### 2️⃣ Run the Streamlit App
```bash
streamlit run app.py
```
This will launch a web app where you can input news text and get predictions.

## 📊 Model Performance
| Metric       | Score  |
|-------------|--------|
| Accuracy    | 98.7%  |
| Precision   | 98.6%  |
| Recall      | 98.9%  |
| F1-Score    | 98.7%  |

## 📌 Future Improvements
- Improve dataset quality and balance
- Experiment with deep learning models
- Deploy the app using **Streamlit Cloud** or **Heroku**

## 🤝 Contributing
Feel free to fork the repo and submit pull requests! 🚀

## 📜 License
This project is licensed under the **MIT License**.

---
Made with ❤️ by **Akhil Danday**

