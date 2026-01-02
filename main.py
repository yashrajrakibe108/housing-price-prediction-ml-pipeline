import os
import joblib
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE = "model.pkl"
Pipeline_FILE = "pipeline.pkl"

def build_pipeline(num_attribs, cat_attribs):
    
    # for numerical columns
    num_Pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
        ])

    # for categorical columns
    cat_pipeline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Construct the full pipeline
    full_pipeline = ColumnTransformer([
        ("num", num_Pipeline, num_attribs),
        ("cat", cat_pipeline, cat_attribs)
    ])
    return full_pipeline

if not os.path.exists(MODEL_FILE):
    #Lets train the model
    housing = pd.read_csv("housing.csv")

# Create a Stratified test set
housing["income_cat"] = pd.cut(housing["median_income"], 
                                        bins = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf], 
                                        labels = [1, 2, 3, 4, 5,])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    housing.loc[test_index].drop("income_cat", axis=1).to_csv("input.csv", index= False)
    housing = housing.loc[train_index].drop("income_cat", axis=1)
    
    
    housing_labels = housing["median_house_value"].copy()
    housing_features = housing.drop("median_house_value", axis =1)


    num_attribs = housing_features.drop("ocean_proximity", axis = 1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    Pipeline = build_pipeline(num_attribs, cat_attribs)
    housing_prepared = Pipeline.fit_transform(housing_features)
    

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared, housing_labels)

    joblib.dump(model, MODEL_FILE)
    joblib.dump(Pipeline, Pipeline_FILE)
    print("Model is trained, Congrats!")

else:
    #Lets do inference
    model = joblib.load(MODEL_FILE)
    Pipeline = joblib.load(Pipeline_FILE)

    input_data = pd.read_csv("input.csv")
    transform_input = Pipeline.transform(input_data)
    predictions = model.predict(transform_input)
    input_data["median_house_value"] = predictions

    input_data.to_csv("output.csv", index=False)
    print("Inference is completed, results save to output.csv Enjoy!")


 