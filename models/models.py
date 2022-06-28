import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
from preprocess.nlp_preprocessing import nlp_preprocess_text
from vectorization.vectorization import TfIdf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import svm
from collections import Counter
import time
from enum import Enum


class ModelType(Enum):
    RANDOM = 1
    SVM = 2
    MNB = 3
    LR = 4
    NN = 5
    # add here more

'''
    creates a classification report (precision, recall, f1-score)
'''
def createClassificationReport(model, x_test, y_test):
    predictions = model.predict(x_test)
    print('THE CLASSIFICATION REPORT: ')
    print(classification_report(y_test, predictions))

    return predictions


'''
    plots a confusion matrix
'''
def displayConfusionMatrix(model, x_test, y_test, classes, titleSuffix : ModelType, normalize=True):
    predictions = model.predict(x_test)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    cm = confusion_matrix(y_test, predictions)
    # plt.figure()

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix ' + titleSuffix.name)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.show()


'''
    creates a model using only the users_df and not the feature matrix, meaning we create
    a model based solely on the text and the user id and preprocess the texts with removal of punctation, 
    lemmatisation and removal of stopwords
    we apply then feature engineering using bag of words to create numerical features out of the text and feed that 
    into a multinomial naive bayes algorithm
    @return: void
'''
def createW2VDeepLearningModel(users_df: pd.DataFrame):
    start = time.time()

    users_df['Body'] = users_df.loc[:, 'Body'].apply(lambda x: nlp_preprocess_text(x))
    print("Found in time [s]: " + str(time.time() - start))

    X_features = users_df['Body']

    cv = CountVectorizer()

    X = cv.fit_transform(X_features)
    y = users_df['ID_User']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    model = MultinomialNB()
    model = model.fit(X_train, y_train)

    print("-" * 21)
    print("TRAINING ACCURACY: ")
    print(model.score(X_train, y_train))

    print("-" * 21)
    print("TESTING VALIDATION ACCURACY: ")
    print(model.score(X_test, y_test))

    predictions = model.predict(X_test)
    print("-" * 21)
    print("PREDICTION ACCURACY IS: ")
    print(accuracy_score(y_test, predictions))


'''
    creates a model using our feature matrix, where we already have generated numerical features for every comment and 
    use then the multinomial naive bayes algorithm to classify the comments
    @return: void
'''
def createModelWithFeatureMatrix(features_matrix: pd.DataFrame, modelType: ModelType, print_report=False, print_cm=False):
    col = 'ID_User'
    y = features_matrix[col]
    X = features_matrix.loc[:, features_matrix.columns != col]
    classes = features_matrix.ID_User.unique()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("-" * 84)
    print("Using",modelType)
    model = None
    if modelType == ModelType.RANDOM:
        model = DummyClassifier(strategy="most_frequent") # baseline model
    elif modelType == ModelType.SVM:
        model = svm.SVC(kernel='poly', degree=2)
    elif modelType == ModelType.MNB:
        model = MultinomialNB()
    elif modelType == ModelType.LR:
        model = LogisticRegression(solver='liblinear', C=10, penalty='l2')
    else:
        # todo implement NN ?
        assert("Unknown type")

    model = model.fit(X_train, y_train)

    print("TRAINING ACCURACY: ", model.score(X_train, y_train))
    print("-" * 21)
    print("TESTING ACCURACY: ", model.score(X_test, y_test))
    print("-" * 21)
    predictions = model.predict(X_test)
    print("PREDICTION ACCURACY: ", accuracy_score(y_test, predictions))
    print("-" * 21)


    if print_report: createClassificationReport(model, X_test, y_test)
    if print_cm: displayConfusionMatrix(model, X_test, y_test, classes, modelType)

