# Telco Customer Churn Prediction App

An end-to-end Machine Learning project designed to predict telecom customer churn using IBM's Telco dataset. The project includes a complete data prep pipeline, model training, and a live user interface built for real-time predictions.

## 🚀 Live Demo
👉 [(https://mohamed-churn-predictor.streamlit.app)]

---

## 📌 Project Overview
Customer churn is one of the most critical metrics for telecom companies. This project focuses on building a robust Machine Learning pipeline that cleans behavioral and demographic customer data, handles feature engineering via custom transformers, and deploys a serialized model to a user-friendly interface.

### Key Features:
* **Custom Transformers:** Built production-grade data cleansing and feature engineering pipelines.
* **Optimized Pipeline:** Serialized the entire workflow (preprocessing + model) into a single `.pkl` file to eliminate data leakage.
* **Interactive UI:** A fully functional web dashboard built with Streamlit for single-customer churn scoring.

---

## 📁 Repository Structure
* `.streamlit/` - Configuration files for Streamlit cloud deployment.
* `Telco Customer Churn.ipynb` - Jupyter notebook containing data exploration, visualization, and model training.
* `UI.py` - Main script powering the Streamlit web application.
* `WA_Fn-UseC_-Telco-Customer-Churn.csv` - The raw IBM Telco Customer dataset.
* `optimized_telecom_churn_model.pkl` - The final, optimized, ready-to-use trained inference pipeline.
* `requirements.txt` - Python dependencies optimized for Linux-based servers.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python 3.11
* **Framework:** Streamlit
* **ML Libraries:** Scikit-Learn, Pandas, NumPy
* **Serialization:** Pickle

---

## ⚙️ Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/mohamed1576/Telco-Customer-Churn.git](https://github.com/mohamed1576/Telco-Customer-Churn.git)
   cd Telco-Customer-Churn

   🧠 Methodology & Pipeline
1. Data Preprocessing: Handled missing values, formatted data types, and managed categorical variables.

2. Feature Engineering: Implemented 2 custom transformers to ensure reproducibility and prevent data leakage during inference.

3. Model Selection & Optimization: Evaluated workflows using cross-validation and hyperparameter tuning to secure the highest predictive accuracy.

4. Deployment: Bundled the pipeline into optimized_telecom_churn_model.pkl for instantaneous real-time inference on the web.
