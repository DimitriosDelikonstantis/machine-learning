import itertools
import pandas as pd
import numpy as np
import matplotlib as plt
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


# Import dataset in a dataframe
# Pandas.read_csv reads a comma-separated values (csv) file into Dataframe and returns a two-dimensional data structure with labeled axes.
Dataframe = pd.read_csv(r'C:\Users\dimde\Documents\University of Piraeus - MSc in Artificial Intelligence\Courses\First semester\Machine learning\Assignments\Machine learning\Fake news\Dataset\train.csv')

# Get the dataframe shape
# Returns a tuple representing the dimensionality of the Dataframe
Dataframe.shape
print(Dataframe.shape)

# Get the dataframe head
# Returns the first and last 5 rows of the Dataframe
Dataframe.head()
print(Dataframe.head)

# Convert the 0, 1 labels to 'REAL' and 'FAKE' for simplicity
# With Dataframe.loc set value for an entire column
Dataframe.loc[(Dataframe['label'] == 1) , ['label']] = 'FAKE'
Dataframe.loc[(Dataframe['label'] == 0) , ['label']] = 'REAL'
print(Dataframe.head)

# Isolate the feature label from the rest of the dataframe
labels = Dataframe.label
labels.head()
print(labels.head)

# Split the dataset
#Test for different case scenarios
# Test 1 -> 60% train, 40% test, random_state=7 -> Accuracy: 95.82%
#x_train,x_test,y_train,y_test=train_test_split(Dataframe['text'].values.astype('str'), labels, test_size=0.4, random_state=7)
# Test 2 -> 65% train, 35% test, random_state=7 -> Accuracy: 96.17%
#x_train,x_test,y_train,y_test=train_test_split(Dataframe['text'].values.astype('str'), labels, test_size=0.35, random_state=7)
# Test 3 -> 70% train, 30% test, random_state=7 -> Accuracy: 95.99%
#x_train,x_test,y_train,y_test=train_test_split(Dataframe['text'].values.astype('str'), labels, test_size=0.3, random_state=7)
# Test 4 -> 75% train, 25% test, random_state=7 -> Accuracy: 96.21%
#x_train,x_test,y_train,y_test=train_test_split(Dataframe['text'].values.astype('str'), labels, test_size=0.25, random_state=7)
# Test 5 -> 80% train, 20% test, random_state=7 -> Accuracy: 96.56%
x_train,x_test,y_train,y_test=train_test_split(Dataframe['text'].values.astype('str'), labels, test_size=0.2, random_state=7)
# Test 6 -> 85% train, 15% test, random_state=7 -> Accuracy: 96.19%
#x_train,x_test,y_train,y_test=train_test_split(Dataframe['text'].values.astype('str'), labels, test_size=0.15, random_state=7)

# Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit & transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

# Initialize the PassiveAggressiveClassifier and fit training sets
pa_classifier=PassiveAggressiveClassifier(max_iter=50)
pa_classifier.fit(tfidf_train,y_train)

# Predict and calculate accuracy
y_pred=pa_classifier.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

# Build confusion matrix
Conf_matrix = confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])
print('Confusion matrix: ' '\n',  Conf_matrix)