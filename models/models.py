import pandas as pd
import matplotlib.pyplot as plt
import itertools
import numpy as np
from preprocess.nlp_preprocessing import preprocess
from vectorization import TfIdf
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import svm
from collections import Counter
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
    elif modelType == 'LR':
        model = LogisticRegression(solver='liblinear', C=10, penalty='l2')

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

    # displayClassReportAndCM(model, X_test, y_test, classes, modelType)


'''
    Function which gets the top nr_authors with the most comments and then we take from the nr_authors the nr_reviews, 
    so that we get a balanced dataset and we use this then to create a LinearSVC model and a tfidfvectorized model
'''


def getTopAuthorComments(users_df: pd.DataFrame, nr_authors: int, nr_reviews: int):
    users_df_list = users_df.to_dict('records')
    top100authors = Counter([user_['ID_User'] for user_ in users_df_list]).most_common(nr_authors)

    keep_ids = {pr[0]: 0 for pr in top100authors}

    keep_comments = []
    for user in users_df_list:
        uid = user['ID_User']
        if uid in keep_ids and keep_ids[uid] < nr_reviews:
            keep_comments.append(user)
            keep_ids[uid] += 1

    texts = [comment['Body'] for comment in keep_comments]
    authors = [comment['ID_User'] for comment in keep_comments]

    vectors = TfIdf(texts, (1, 5), 3, False)
    print('Vectors shape from TFIDF: ')
    print(vectors.shape)
    X_train, X_test, y_train, y_test = train_test_split(vectors, authors, test_size=0.2, random_state=42)

    '''
        case for Linearsvc model
    '''
    svm_ = LinearSVC()
    svm_.fit(X_train, y_train)

    predictions = svm_.predict(X_test)
    print('-' * 42)
    print('Result for Model with top ' + str(nr_authors) + ' authors and their top ' + str(nr_reviews) +
          ' comments Linear SVC is: ')
    print(accuracy_score(y_test, predictions))
    print('-' * 42)
