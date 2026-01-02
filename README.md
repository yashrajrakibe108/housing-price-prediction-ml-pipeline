Housing Price Prediction – ML Pipeline

A production-ready Machine Learning project that predicts housing prices using an end-to-end Scikit-learn pipeline. The project demonstrates data preprocessing, stratified sampling, model training, evaluation, artifact persistence, and batch inference with a clear path to deployment.

Objective

To build a reliable and reusable Machine Learning pipeline that predicts median house values from structured housing data while following industry-standard ML practices suitable for deployment.

Key Highlights

End-to-end ML workflow (training → evaluation → inference)

Modular preprocessing using Scikit-learn Pipelines

Stratified train-test split based on income categories

Random Forest regression model

Cross-validation for performance evaluation

Persisted model and preprocessing pipeline

Deployment-ready project structure

ML Pipeline Architecture

Data Ingestion
→ Stratified Sampling
→ Feature Engineering
→ Preprocessing Pipeline
→ Model Training
→ Model Evaluation
→ Model Persistence
→ Batch Inference

Preprocessing Strategy

Numerical Features

Median imputation for missing values

Standard scaling

Categorical Features

One-hot encoding

Safe handling of unseen categories

All preprocessing steps are encapsulated in a reusable pipeline to ensure consistency between training and inference.

Model Details

Algorithm: Random Forest Regressor

Evaluation Metric: Root Mean Squared Error (RMSE)

Validation Strategy: K-Fold Cross-Validation

The trained model is serialized and reused for inference to avoid retraining.

Project Structure

housing.csv
Raw housing dataset

test.csv
Stratified test dataset used for inference

model.pkl
Serialized trained model

pipeline.pkl
Serialized preprocessing pipeline

output.csv
Prediction results

main script
Handles training and inference logic

Training & Inference Logic

If no trained model is found, the pipeline automatically trains a new model and saves artifacts

If a trained model exists, the pipeline performs inference on unseen data

Predictions are exported to a CSV file

This design mirrors real-world ML deployment workflows.

Deployment Readiness

This project is designed to be easily deployed using:

Flask / FastAPI for REST APIs

Batch inference pipelines

Cloud platforms (AWS, GCP, Azure)

Containerization using Docker

The separation of preprocessing and modeling ensures safe and repeatable predictions in production.

Use Cases

Real estate price estimation

Learning production-grade ML pipelines

Portfolio project for Data Scientist / ML Engineer roles

Base template for ML model deployment

Future Enhancements

Hyperparameter optimization

Model versioning

API-based real-time inference

Dockerization

CI/CD integration

Monitoring and logging

Author

Yashraj Rakibe
AI / ML Engineer | Data Scientist
Python | Machine Learning | Data Science
