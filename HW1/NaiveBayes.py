import pandas as pd
import numpy as np
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Using Multinomial NB since it gave the best results.
# Gaussian NB was pretty bad, around 80% testing accuracy.

# Loading datasets and labels.
X_train = pd.read_csv('./ReutersData/train.csv', header=None).values
X_test  = pd.read_csv('./ReutersData/test.csv', header=None).values
y_train = np.loadtxt('./ReutersData/train_labels.txt', dtype=int)
y_test  = np.loadtxt('./ReutersData/test_labels.txt', dtype=int)

# Initializing the classifier.
nb_classifier = MultinomialNB()

# Training the classifier and measuring time.
start_time = time.perf_counter()
nb_classifier.fit(X_train, y_train)
training_time = time.perf_counter() - start_time

# Predicting on training and testing data.
y_pred_train = nb_classifier.predict(X_train)
y_pred_test  = nb_classifier.predict(X_test)

# Calculating accuracy.
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy  = accuracy_score(y_test, y_pred_test)

# Outputting results.
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")
print(f"Training Time: {training_time:.4f} seconds")