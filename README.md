🏠 Housing Price Prediction – ML Pipeline

An end-to-end Machine Learning project that predicts housing prices using structured data.
This project demonstrates a complete ML workflow including data preprocessing, stratified sampling, model training, cross-validation, model persistence, and inference using Scikit-learn pipelines.

📌 Project Overview

The goal of this project is to build a robust and reusable ML pipeline for predicting median house values based on housing-related features.
It follows industry best practices such as feature engineering, pipeline abstraction, and separation of training and inference logic.

🛠️ Tech Stack

Python

Pandas, NumPy

Scikit-learn

Joblib

📂 Project Structure
housing-price-prediction-ml-pipeline/
│
├── housing.csv              # Raw dataset
├── model.pkl                # Trained RandomForest model
├── pipeline.pkl             # Preprocessing pipeline
├── test.csv                 # Stratified test dataset
├── output.csv               # Predictions generated during inference
├── main.py                  # Training & inference script
└── README.md                # Project documentation

🔄 Workflow

Data Loading

Load housing dataset from CSV

Stratified Sampling

Create income categories

Perform stratified train-test split

Data Preprocessing

Numerical features: median imputation + standard scaling

Categorical features: one-hot encoding

Implemented using ColumnTransformer and Pipeline

Model Training

RandomForest Regressor

5-fold cross-validation using RMSE metric

Model Persistence

Save trained model and preprocessing pipeline using joblib

Inference

Load saved model and pipeline

Generate predictions on unseen data

Save results to output.csv

📊 Model Evaluation

Algorithm: RandomForest Regressor

Metric: Root Mean Squared Error (RMSE)

Validation: 5-Fold Cross Validation

Cross-validation ensures the model generalizes well and avoids overfitting.

▶️ How to Run
1️⃣ Install Dependencies
pip install pandas numpy scikit-learn joblib

2️⃣ Train the Model
python main.py


If model.pkl does not exist → training starts

Model and pipeline are saved after training

3️⃣ Run Inference
python main.py


If model.pkl exists → inference runs

Predictions saved to output.csv

🚀 Key Highlights

End-to-end ML pipeline

Clean separation of preprocessing and model logic

Stratified sampling for better data distribution

Reusable and production-friendly design

Resume and portfolio ready project

📈 Future Improvements

Add Flask / FastAPI REST API

Hyperparameter tuning

Logging and exception handling

Dockerization

Model monitoring

👤 Author

Yashraj Rakibe
AI / ML Engineer | Data Science Enthusiast
