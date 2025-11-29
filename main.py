# Goal is to create a model using the data to predict my surviviabilty on the titanic based off of the data point dropped in
# DATA
import pandas as pd
# Load and creating data frame
import numpy as np

# MACHINE LEARNING
# Takes list of data and converts it into an array, model understandds data better
from sklearn.model_selection import train_test_split, GridSearchCV
# Allows us to create training set and testing set, test multiple scenarios
from sklearn.preprocessing import MinMaxScaler
# Cleans data
from sklearn.neighbors import KNeighborsClassifier
# Model we will be using
from sklearn.metrics import accuracy_score,confusion_matrix
# Metrics to test how our model is performing

# VISUALISATION
import matplotlib.pyplot as plt
import seaborn as sns

# Creating a data frame
data = pd.read_csv("titanic.csv")
data.info()
# Seeing if you need to clean the data
print(data.isnull().sum())
# Machine learning model only understands numbers

# DATA CLEANING AND FEATURE ENGINEERING (Adding new coloums to enhance model)
def preprocess_data(data_frame):
  # Removing irrelevant data
  data_frame.drop(columns=["PassengerId", "Name", "Ticket", "Cabin", "Embarked"], inplace=True)
  fill_missing_ages(data_frame)
  data_frame["Fare"].fillna(data_frame["Fare"].median(), inplace=True)

  # Convert gender
  data_frame["Sex"] = data_frame["Sex"].map({'male':1, 'female':0})
  
  # Feature engineering
  data_frame["FamilySize"] = data_frame["SibSp"] + data_frame["Parch"]
  # Tells us if something is where it should be
  data_frame["isAlone"] = np.where(data_frame["FamilySize"] == 0, 1, 0)
  data_frame["FareBin"] = pd.qcut(data_frame["Fare"], 4, labels=False)
  data_frame["AgeBin"] = pd.cut(data_frame["Age"], bins=[0,12,20,40,60, np.inf], labels=False)
  data_frame.fillna(0, inplace=True)
  return data_frame

# FILL IN MISSING AGES
# Getting avergae age of the class that that person is in

def fill_missing_ages(df):

  age_fill_map = {}
  for pclass in df["Pclass"].unique():
    if pclass not in age_fill_map:
      age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].median()
      
  df["Age"] = df.apply(lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"], axis=1)
  
data = preprocess_data(data)
# Creature features and target vars, the answer

x = data.drop(columns=["Survived"])
y = data["Survived"]

# Function returns 4 vals. Think of these as flashcards, X is the front of the flashcard, y is the back of the flashcard.
# Computer is going to look at the x's and y's training to learn. It is allowed to see and back while learning.
# When testing, will only be allowed to look at the front of the flashcard
x_train, x_test, y_train, y_test = train_test_split(x , y , test_size=0.25, random_state=42)

# ML PREPROCESSING - Taking our data and making sure everything is numerically formatted for the model to understand
scaler = MinMaxScaler()

# This takes the training data
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# HYPERPARAMETER TUNING - get the best model possible
# KNN Model. Chart with scatter plot. Represnts ppl in our data set. Our model is gonna drop a new datapoont in the set then collect information from the nearest datapoint. Its neighbours.
def tune_model(x_train, y_train):
  param_grid = {
    "n_neighbors" : range(1, 21),
    "metric" : ["euclidean", "manhattan", "minkowski"],
    "weights":["uniform", "distance"]
  }
  
  model = KNeighborsClassifier()
  grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)
  # Fit is the function that allows us to train the data
  grid_search.fit(x_train, y_train)
  return grid_search.best_estimator_

best_model = tune_model(x_train, y_train)

# PREDICTIONS AND EVALUATE

def evaluate_model(model, x_test, y_test):
  prediction = model.predict(x_test)
  accuracy = accuracy_score(y_test, prediction)
  matrix = confusion_matrix(y_test, prediction)
  return accuracy, matrix

accuracy, matrix = evaluate_model(best_model, x_test, y_test)

# print(data.isnull().sum())

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Confusion Matrix:')
print(matrix)

def plot_model(matrix):
  plt.figure(figsize=(10,7))
  sns.heatmap(matrix, annot=True, fmt="d", xticklabels=["Survived", "Not Survived"], yticklabels=["Not Survived", "Survived"])
  plt.title("Confusion matrix")
  plt.xlabel("Predicted Value")
  plt.ylabel("True Values")
  plt.show()
  
plot_model(matrix)