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

# DATA CLEANING