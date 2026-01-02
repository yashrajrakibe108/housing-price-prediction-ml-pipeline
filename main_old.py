
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score
# 1. Loading the Dataset
housing = pd.read_csv("housing.csv")

# 2. Create a Stratified test set
housing["income_cat"] = pd.cut(housing["median_income"], 
                                        bins = [0.0, 1.5, 3.0, 4.5, 6.0, np.inf], 
                                        labels = [1, 2, 3, 4, 5,])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index].drop("income_cat", axis=1)
    strat_test_set = housing.loc[test_index].drop("income_cat", axis=1)

# we will work on the copy of training data
housing = strat_train_set.copy()

# 3. Separate features and labels 
housing_labels = housing["median_house_value"].copy()
housing = housing.drop("median_house_value", axis =1)

#print(housing_labels, housing)

# 4. Seperate numerical and categorical columns
num_attribs = housing.drop("ocean_proximity", axis = 1).columns.tolist()
cat_attribs = ["ocean_proximity"]

# 5. Making the pipeline 
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

# 6. Transform the data
housing_prepared = full_pipeline.fit_transform(housing)
print(housing_prepared.shape)

# 7. Train the model

#linear Regression
lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
lin_preds = lin_reg.predict(housing_prepared)
lin_rmse = root_mean_squared_error(housing_labels, lin_preds)
#print(f"Root mean squared error for linear Regression is {lin_rmse}")
lin_rmes = -cross_val_score(lin_reg, housing_prepared, housing_labels,
                            scoring="neg_root_mean_squared_error", cv=10)

print(pd.Series(lin_rmes).describe())


#Decision Tree 
tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
tree_preds = tree_reg.predict(housing_prepared)
tree_rmse = root_mean_squared_error(housing_labels, tree_preds)
#print(f"Root mean squared error for Decision Tree Regression is {tree_rmse}")
tree_rmes = -cross_val_score(tree_reg, housing_prepared, housing_labels,
                             scoring="neg_root_mean_squared_error", cv=10)

print(pd.Series(tree_rmes).describe())

#Random Foresest Regression
Random_Foresest_reg = RandomForestRegressor()
Random_Foresest_reg.fit(housing_prepared, housing_labels)
Random_Foresest_preds = Random_Foresest_reg.predict(housing_prepared)
Random_Foresest_rmse = root_mean_squared_error(housing_labels, Random_Foresest_preds)
#print(f"Root mean squared error for RandomForesest Regression is {Random_Foresest_rmse}")
Random_Foresest_rmes = -cross_val_score(Random_Foresest_reg, housing_prepared, housing_labels,
                                        scoring="neg_root_mean_squared_error", cv=10)

print(pd.Series(Random_Foresest_rmes).describe())
