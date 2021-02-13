import itertools
import pandas as pd
import numpy as np
from matplotlib import pyplot
from sklearn import svm
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.linear_model import PassiveAggressiveClassifier, LogisticRegression, Perceptron
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_curve, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc


# Import dataset in a dataframe
# Pandas.read_csv reads a comma-separated values (csv) file into dataframe and returns a two-dimensional data structure with labeled axes.
dataframe = pd.read_csv(r'C:\Users\dimde\Documents\University of Piraeus - MSc in Artificial Intelligence\Courses\First semester\Machine learning\Assignments\Machine learning\Fake news\Dataset\train.csv')

# Get dataframe information
# Represents the dimensionality of the dataframe
print('Dataframe shape: ', dataframe.shape, '\n')
# Represents the axes of the dataframe
print('Dataframe axes: ', dataframe.axes, '\n')
# Returns the dtypes in the dataframe
print('Dataframe dtypes: ', dataframe.dtypes, '\n')
# Returns an int representing the number of elements in the dataframe
print('Dataframe size: ', dataframe.size, '\n')
# Returns the first 10 rows of the dataframe
print('Dataframe head: ', '\n', dataframe.head(10), '\n')

# Convert the 0, 1 labels to 'REAL' and 'FAKE' for simplicity
# With Dataframe.loc set value for an entire column
dataframe.loc[(dataframe['label'] == 1) , ['label']] = 'FAKE'
dataframe.loc[(dataframe['label'] == 0) , ['label']] = 'REAL'

# Visualize the head again to check the label changes
print('Converted dataframe head: ', '\n', dataframe.head(10), '\n')

# Isolate the text and label feature columns from the rest of the dataframe
labels = dataframe.label
labels.head()
text = dataframe.text
text.head()
print(labels.head)
print(text.head)


# Split the dataset
# Test for different case scenarios
# Test 1 -> 60% train, 40% test, random_state = 7 -> Accuracy: 95.82%
#x_train,x_test,y_train,y_test = train_test_split(text.values.astype('str'), labels, test_size = 0.4, random_state = 7)
# Test 2 -> 65% train, 35% test, random_state = 7 -> Accuracy: 96.17%
#x_train,x_test,y_train,y_test = train_test_split(text.values.astype('str'), labels, test_size = 0.35, random_state = 7)
# Test 3 -> 70% train, 30% test, random_state = 7 -> Accuracy: 95.99%
#x_train,x_test,y_train,y_test = train_test_split(text.values.astype('str'), labels, test_size = 0.3, random_state = 7)
# Test 4 -> 75% train, 25% test, random_state = 7 -> Accuracy: 96.21%
#x_train,x_test,y_train,y_test = train_test_split(text.values.astype('str'), labels, test_size = 0.25, random_state = 7)
# Test 5 -> 80% train, 20% test, random_state = 7 -> Accuracy: 96.56%
x_train,x_test,y_train,y_test = train_test_split(text.values.astype('str'), labels, test_size = 0.2, random_state = 7)
# Test 6 -> 85% train, 15% test, random_state = 7 -> Accuracy: 96.19%
#x_train,x_test,y_train,y_test = train_test_split(text.values.astype('str'), labels, test_size = 0.15, random_state = 7)

# Custom stop words list
stop_words_list = ['a, the, of, this, that']

# Initialize a TfidfVectorizer
# Test for different case scenarios
tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7, max_features = 100)
#tfidf_vectorizer = TfidfVectorizer(stop_words = stop_words_list, max_df = 0.7)
#tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7, ngram_range= (2, 2))
#tfidf_vectorizer = TfidfVectorizer(max_df = 0.7, ngram_range= (2, 2))
#tfidf_vectorizer = TfidfVectorizer(max_df = 0.5, max_features = 1000, ngram_range= (2, 2))
#tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.7, ngram_range= (2, 2))
#tfidf_vectorizer = TfidfVectorizer(stop_words = 'english', max_df = 0.6, ngram_range= (3, 3))

# Fit and transform train set and test set to TfidfVectorizer
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# View vocabulary length
vocabulary_len = len(tfidf_vectorizer.get_feature_names())
print('TfidfVectorizer number of features: ', vocabulary_len, '\n')

# Visualize the stop words used by TfidfVectorizer
stop_words_len = tfidf_vectorizer.get_stop_words()
print('TfidfVectorizer stop words: ', '\n', stop_words_len, '\n')


# Initialize the PassiveAggressiveClassifier and fit training sets
#pa_classifier = PassiveAggressiveClassifier(max_iter = 50)
#pa_classifier.fit(tfidf_train, y_train)
# Predict with Passive aggressive classifier
#y_pred = pa_classifier.predict(tfidf_test)

# Initialize the Multinomial Naive Bayes classifier and fit training sets
#multinomialnb_classifier = MultinomialNB()
#multinomialnb_classifier.fit(tfidf_train, y_train)
# Get probability estimates for the x_test
#cl_probs = multinomialnb_classifier.predict_proba(tfidf_test)
#cl_probs = cl_probs[:, 1]
# Predict with Multinomial Naive Bayes classifier
#y_pred = multinomialnb_classifier.predict(tfidf_test)

# Initialize the SVC classifier and fit training sets
#svc_classifier = svm.SVC(probability = True)
#svc_classifier.fit(tfidf_train, y_train)
# Get probability estimates for the x_test
#cl_probs = svc_classifier.predict_proba(tfidf_test)
#cl_probs = cl_probs[:, 1]
# Predict with SVC classifier
#y_pred = svc_classifier.predict(tfidf_test)

# Initialize the LogisticRegression classifier and fit training sets
#lr_classifier = LogisticRegression()
#lr_classifier.fit(tfidf_train, y_train)
# Get probability estimates for the x_test
#cl_probs = lr_classifier.predict_proba(tfidf_test)
#cl_probs = cl_probs[:, 1]
# Predict with Logistic Regression classifier
#y_pred = lr_classifier.predict(tfidf_test)

# Initialize the Perceptron classifier and fit training sets
#perceptron_classifier = Perceptron()
#perceptron_classifier.fit(tfidf_train, y_train)
# Get probability estimates for the x_test
#cl_probs = perceptron_classifier.predict_proba(tfidf_test)
#cl_probs = cl_probs[:, 1]
# Predict with Logistic Regression classifier
#y_pred = perceptron_classifier.predict(tfidf_test)

# # Initialize the KNeighborsClassifier and fit training sets
# knn_classifier = KNeighborsClassifier()
# knn_classifier.fit(tfidf_train, y_train)
# Get probability estimates for the x_test
# cl_probs = knn_classifier.predict_proba(tfidf_test)
# cl_probs = cl_probs[:, 1]
# # Predict with KNeighborsClassifier
# y_pred = knn_classifier.predict(tfidf_test)

# # Initialize the DecisionTreeClassifier and fit training sets
# Decision_tree_classifier = DecisionTreeClassifier()
# Decision_tree_classifier.fit(tfidf_train, y_train)
# # Get probability estimates for the x_test
# cl_probs = Decision_tree_classifier.predict_proba(tfidf_test)
# cl_probs = cl_probs[:, 1]
# # Predict with DecisionTreeClassifier
# y_pred = Decision_tree_classifier.predict(tfidf_test)

# Initialize the LinearDiscriminantAnalysis and fit training sets
lda_classifier = LinearDiscriminantAnalysis()
lda_classifier.fit(tfidf_train.toarray(), y_train)
# Get probability estimates for the x_test
cl_probs = lda_classifier.predict_proba(tfidf_test)
cl_probs = cl_probs[:, 1]
# Predict with LinearDiscriminantAnalysis
y_pred = lda_classifier.predict(tfidf_test)


# Metrics
# Calculate accuracy: (tp + tn)/(tp + tn + fp + fn)
Accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ', Accuracy * 100, '%', '\n')

# Calculate confusion matrix
#           Actual class
# Predicted [TP, FP]
# class     [FN, TN]
Conf_matrix = confusion_matrix(y_test, y_pred, labels = ['FAKE', 'REAL'])
# Extract true negatives, false positives, false negatives and true positives from confusion matrix
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
print('Confusion matrix: ' '\n', Conf_matrix, '\n')

# Calculate precision: tp / (tp + fp)
precision = tp / (tp + fp)
print('Precision: ', precision, '\n')

# Calculate recall: tp / (tp + fn)
# True positive rate / Sensitivity
recall = tp / (tp + fn)
print('Recall: ', recall, '\n')

# Calculate specificity: Tn / (Tn + Fp)
specificity = tn / (tn+fp)
print('Specificity: ', specificity, '\n')

#Calculate False positive rate
False_positive_rate = 1 - specificity
print('False positive rate: ', False_positive_rate, '\n')

# Calculate F1-score
f1_score = 2 * (precision * recall) / (precision + recall)
print('F1-score: ', f1_score, '\n')

# Calculate the ROC curve
ns_probs = [0 for _ in range(len(y_test))]
ns_auc = roc_auc_score(y_test, ns_probs)
cl_auc = roc_auc_score(y_test, cl_probs)
print('Random: ROC AUC =', ns_auc)
print('Classifier: ROC AUC =', cl_auc)
ns_fpr, ns_tpr, _ = roc_curve(y_test, ns_probs, pos_label = 'REAL')
cl_fpr, cl_tpr, _ = roc_curve(y_test, cl_probs, pos_label = 'REAL')
# Plot the ROC curve
pyplot.plot(ns_fpr, ns_tpr, linestyle = '--', label = 'Random')
pyplot.plot(cl_fpr, cl_tpr, marker = '.', label = 'Classifier')
# Name axis labels
pyplot.xlabel('False Positive Rate')
pyplot.ylabel('True Positive Rate')
# Show the legend
pyplot.legend()
# Show the plot
pyplot.show()

# Calculate the precision-recall curve
cl_precision, cl_recall, _ = precision_recall_curve(y_test, cl_probs, pos_label = 'REAL')
cl_auc = auc(cl_recall, cl_precision)
print('Classifier: auc= ', cl_auc)
# Plot the precision-recall curves
random = len(y_test[y_test == 1]) / len(y_test)
pyplot.plot([0, 1], [random, random], linestyle='--', label = 'Random')
pyplot.plot(cl_recall, cl_precision, marker='.', label = 'Classifier')
# Name axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# Show the legend
pyplot.legend()
# Show the plot
pyplot.show()
