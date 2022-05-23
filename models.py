import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
from preprocessing import preprocess
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import svm
import time

'''
    creates a classification report (precision, recall, f1-score)
'''


def createClassificationReport(model, x_test, y_test):
    predictions = model.predict(x_test)
    print('-' * 21)
    print('THE CLASSIFICATION REPORT: ')
    print(classification_report(y_test, predictions))

    return predictions


'''
    plots a confusion matrix
'''


def displayClassReportAndCM(model, x_test, y_test, classes, titleSuffix, normalize=True):
    # predictions = createClassificationReport(model, x_test, y_test)
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
    plt.title('Confusion Matrix ' + titleSuffix)
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


def createModelWihoutFeatureMatrix(users_df: pd.DataFrame):

    start = time.time()

    users_df['Body'] = users_df.loc[:, 'Body'].apply(lambda x: preprocess(x))
    print("Found in time [s]: " + str(time.time() - start))

    X = users_df['Body']
    y = users_df['ID_User']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    bow_transformer = CountVectorizer(analyzer=preprocess).fit(X_train)
    # transforming into Bag-of-Words and hence textual data to numeric..
    text_bow_train = bow_transformer.transform(X_train)  # ONLY TRAINING DATA
    # transforming into Bag-of-Words and hence textual data to numeric..
    text_bow_test = bow_transformer.transform(X_test)  # TEST DATA


    model = MultinomialNB()
    model = model.fit(text_bow_train, y_train)

    print("-" * 21)
    print("TRAINING ACCURACY: ")
    print(model.score(text_bow_train, y_train))

    print("-" * 21)
    print("TESTING VALIDATION ACCURACY: ")
    print(model.score(text_bow_test, y_test))

    predictions = model.predict(X_test)
    print("-" * 21)
    print("PREDICTION ACCURACY IS: ")
    print(accuracy_score(y_test, predictions))


'''
    creates a model using our feature matrix, where we already have generated numerical features for every comment and 
    use then the multinomial naive bayes algorithm to classify the comments
    @return: void
'''


def createModelWithFeatureMatrix(features_matrix: pd.DataFrame, modelType: str):
    col = 'ID_User'
    y = features_matrix[col]
    X = features_matrix.loc[:, features_matrix.columns != col]
    classes = features_matrix.ID_User.unique()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if modelType == 'SVM':
        model = svm.SVC(kernel='poly', degree=2)
    elif modelType == 'MNB':
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

    displayClassReportAndCM(model, X_test, y_test, classes, modelType)