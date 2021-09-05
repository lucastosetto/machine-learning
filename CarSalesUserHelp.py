############################################################################

# Project Name: Car Sales User Help
# Version: 1.0.0
# Author: Lucas Tosetto Morvillo
# Date: 05/09/2021

############################################################################

# Project Description:

# This application is a simplified version of what could be used in car
# sales websites to help users define their car prices.

# It uses previous sales data (car's mileage per year, model year and 
# annouced price) and a Decision Tree Classifier from Scikit Learn library 
# to build an engine that classifies wether user input price is suitable or 
# not to lead to a car sale.

############################################################################

import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz

# Load dataset

dataset_url = "https://gist.githubusercontent.com/guilhermesilveira/4d1d4a16ccbf6ea4e0a64a38a24ec884/raw/afd05cb0c796d18f3f5a6537053ded308ba94bf7/car-prices.csv"
dataset = pd.read_csv(dataset_url)

# Preprocessing dataset

dataset['km'] = np.floor(dataset.mileage_per_year * 1.60934).astype(int)

current_year = datetime.today().year
dataset['age'] = current_year - dataset.model_year

class_to_rename = {'yes' : 1, 'no': 0}
dataset.sold = dataset.sold.map(class_to_rename)

del dataset['Unnamed: 0']
del dataset['mileage_per_year']
del dataset['model_year']

dataset = dataset[['age', 'km', 'price', 'sold']]

# Visualizing dataset summary

print('Previous Car Sales Dataset - Summary: \n')
print(dataset.head())

# Create train and test datasets (Train = 75% / Test = 25%)

X = dataset[['age', 'km', 'price']]
y = dataset[['sold']]

np.random.seed(5)
X_train, X_test, raw_y_train, raw_y_test = train_test_split(X, y, test_size=0.25, stratify=y)

# Convert y datasets into 1d arrays

y_train = raw_y_train.values.ravel()
y_test = raw_y_test.values.ravel()

# Define accuracy baseline with dummy classifier

dummy = DummyClassifier()
dummy.fit(X_train, y_train)
baseline = dummy.score(X_train, y_train)

baseline_percentage = 100 * baseline
print(" \n Dummy Classifier Accuracy = %.2f %% \n" % baseline_percentage)

# Decision Tree Classifier Model training and testing

model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

accuracy_percentage = 100 * accuracy
print("Decision Tree Classifier Accuracy = %.2f %%" % accuracy_percentage)

# Visualize Decision Tree

features = X.columns
dot_data = export_graphviz(model, out_file=None, filled=True, rounded=True, feature_names=features, class_names=['no', 'yes'])
plot = graphviz.Source(dot_data)
plot.view()