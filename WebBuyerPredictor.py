############################################################################

# Project Name: Web Buyer Predictor
# Version: 1.0.0
# Author: Lucas Tosetto Morvillo
# Date: 04/09/2021

############################################################################

# Project Description:

#   A sample product website contains three pages:

#       1. Home
#       2. How it Works
#       3. Contact

#   Based on data about which pages previous buyers have entered,
#   this application will predict wether a new visitor is a potential buyer
#   using a Linear Support Vector Machine Classifier from Scikit Learn library.

############################################################################

import pandas as pd
import math as m
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load dataset
dataset_url = "https://gist.githubusercontent.com/guilhermesilveira/2d2efa37d66b6c84a722ea627a897ced/raw/10968b997d885cbded1c92938c7a9912ba41c615/tracking.csv"
dataset = pd.read_csv(dataset_url)

# Split dataset into data inputs and outputs
X = dataset[['home', 'how_it_works', 'contact']]
y = dataset[['bought']]

# Split datasets intro train and test datasets (Train = 75% / Test = 25%)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=20, test_size=0.25, stratify=y)

# Model Training
model = LinearSVC()
model.fit(X_train, y_train.values.ravel())

# Model Testing
predictions = model.predict(X_test)

# Calculate and print model accuracy
svc_accuracy = 100 * accuracy_score(y_test, predictions)
print("Linear SVC Accuracy = %.2f %%" % svc_accuracy)