import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
from vectorization.vectorization import createListOfListOfUnigram
from vectorization.feature_engineering import createPaddingSequence
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from vectorization.feature_matrix import VectorizationType
from sklearn import svm
import time
from enum import Enum
from models import NN


class ModelType(Enum):
    RANDOM = 1
    SVM = 2
    MNB = 3
    LR = 4
    MLP = 5
    XG = 6
    KNN = 7
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


def displayConfusionMatrix(model, x_test, y_test, classes, titleSuffix: ModelType, normalize=True):
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
    X = users_df['cleaned'].values
    y = users_df['ID_User'].values
    train_comments, test_comments, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    [x_train, x_test, voc_size] = createPaddingSequence(train_comments, test_comments)

    model = NN.createConvolutionalModel(voc_size)

    history = model.fit(x_train, y_train, epochs=10, batch_size=10, verbose=1, validation_data=(x_test, y_test))

    loss, accuracy = model.evaluate(x_train, y_train, verbose=1)
    print("Training Accuracy: {:.4f}".format(accuracy))
    loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
    print("Testing Accuracy:  {:.4f}".format(accuracy))


'''
    creates a model using our feature matrix, where we already have generated numerical features for every comment and 
    use then the multinomial naive bayes algorithm to classify the comments
    @return: void
'''


def createModelWithFeatureMatrix(X_train, X_test, y_train, y_test, modelType: ModelType, print_report=False, print_cm=False):
    print("-" * 84)
    print("Using", modelType)
    model = None
    if modelType == ModelType.RANDOM:
        model = DummyClassifier(strategy="most_frequent")  # baseline model
    elif modelType == ModelType.SVM:
        model = svm.SVC(kernel='poly', degree=2)
    elif modelType == ModelType.MNB:
        model = MultinomialNB()
    elif modelType == ModelType.LR:
        model = LogisticRegression(solver='liblinear', C=10, penalty='l2')
    elif modelType == ModelType.MLP:
        model = MLPClassifier(random_state=1, solver="adam", hidden_layer_sizes=(12, 12, 12), activation="relu",
                              early_stopping=True,
                              n_iter_no_change=1)
    elif modelType == ModelType.XG:
        model = XGBClassifier(random_state=42, seed=2, colsample_bytree=0.6, subsample=0.7)
        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
    elif modelType == ModelType.KNN:
        model = KNeighborsClassifier(n_neighbors=7)
    else:
        assert ("Unknown type")

    model = model.fit(X_train, y_train)

    print("TRAINING ACCURACY: ", model.score(X_train, y_train))
    print("-" * 21)
    print("TESTING ACCURACY: ", model.score(X_test, y_test))
    print("-" * 21)
    predictions = model.predict(X_test)
    print("PREDICTION ACCURACY: ", accuracy_score(y_test, predictions))
    print("-" * 21)

    if print_report: createClassificationReport(model, X_test, y_test)
    # if print_cm: displayConfusionMatrix(model, X_test, y_test, classes, modelType)
