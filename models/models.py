import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import svm
from enum import Enum

from vectorization.feature_matrix import VectorizationType
from vectorization.vectorization import getVectorizer


class ModelType(Enum):
    RANDOM = 1
    SVC = 2
    MNB = 3
    LR = 4
    NN = 5
    MLP = 6
    KNN = 7



def createClassificationReport(model, x_test, y_test):
    """
        creates a classification report (precision, recall, f1-score)
    """
    predictions = model.predict(x_test)
    print('THE CLASSIFICATION REPORT: ')
    print(classification_report(y_test, predictions))

    return predictions


def createNNModel(input_dimension: int):
    model = Sequential()
    model.add(Dense(12, input_dim=input_dimension, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    model.compile(loss="binary_crossentropy", optimizer='adam', metrics=["accuracy"])
    model.summary()
    return model


def displayConfusionMatrix(model, x_test, y_test, classes, titleSuffix: ModelType, typeOfFeatureExtraction: str):
    """
        plots a confusion matrix
    """
    predictions = model.predict(x_test)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(15, 15))

    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix ' + titleSuffix.name + ' ' + typeOfFeatureExtraction)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=9)
    plt.yticks(tick_marks, classes, fontsize=9)

    fmt = '.2f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", fontsize=8,
                 color="white" if cm[i, j] > thresh else "black")
    plt.show()


def createModelWithFeatureMatrix(features_matrix: pd.DataFrame, modelType: ModelType, vecType: VectorizationType,
                                 print_report=False,
                                 print_cm=False):
    """
        creates a model using our feature matrix, where we already have generated numerical features for every comment
        and use then the multinomial naive bayes algorithm to classify the comments
        @return: void
    """
    col = 'ID_User'
    y = features_matrix[col]
    X = features_matrix.loc[:, features_matrix.columns != col]
    if vecType == modelType.NN:
        X = (X - X.mean()) / X.std()
    classes = features_matrix.ID_User.unique()

    if vecType == VectorizationType.TfIdf or vecType == VectorizationType.BagOfWords:
        X = features_matrix['Body']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        vectorizer = getVectorizer(vecType)
        vectorizer.fit(X_train.apply(lambda x: np.str_(x)))
        X_train = vectorizer.transform(X_train.apply(lambda x: np.str_(x)))
        X_test = vectorizer.transform(X_test.apply(lambda x: np.str_(x)))
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("-" * 84)
    print("Using", modelType)
    model = None
    if modelType == ModelType.RANDOM:
        model = DummyClassifier(strategy="most_frequent")  # baseline model
    elif modelType == ModelType.SVC:
        model = svm.SVC(kernel='poly', degree=2, random_state=42)
    elif modelType == ModelType.MNB:
        model = MultinomialNB()
    elif modelType == ModelType.LR:
        model = LogisticRegression(solver='sag', C=10, penalty='l2', random_state=42)
    elif modelType == ModelType.NN:
        input_dimension = X_train.shape[1]
        if vecType == VectorizationType.TfIdf or vecType == VectorizationType.BagOfWords:
            X_train = X_train.toarray()
        model = createNNModel(input_dimension)
    elif modelType == ModelType.MLP:
        model = MLPClassifier(random_state=1, solver="adam", hidden_layer_sizes=(12, 12, 12), activation="relu",
                              early_stopping=True,
                              n_iter_no_change=1)
    elif modelType == ModelType.KNN:
        model = KNeighborsClassifier(n_neighbors=7)
    else:
        assert "Unknown type"

    if modelType == ModelType.NN:
        history = model.fit(X_train, y_train, epochs=100, verbose=1, validation_data=(X_test, y_test), batch_size=10)

        loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
        print("Training Accuracy: {:.4f}".format(accuracy))
        loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
        print("Testing Accuracy:  {:.4f}".format(accuracy))
    else:
        model = model.fit(X_train, y_train)

        print("TRAINING ACCURACY: ", model.score(X_train, y_train))
        print("-" * 21)
        print("TESTING ACCURACY: ", model.score(X_test, y_test))
        print("-" * 21)
        predictions = model.predict(X_test)
        print("PREDICTION ACCURACY: ", accuracy_score(y_test, predictions))
        print("-" * 21)

    if print_report: createClassificationReport(model, X_test, y_test)
    if print_cm: displayConfusionMatrix(model, X_test, y_test, classes, modelType, vecType.name)
