from typing import List

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import gensim
import nltk
from gensim.models import Word2Vec
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

nltk.download('stopwords')
nltk.download('punkt')

'''
    Helper Function to get list of lists of n-grams
'''


def createListOfListOfUnigram(corpus) -> List[List[str]]:
    lst_corpus = []
    for string in corpus:
        lst_words = string.split()
        lst_grams = [" ".join(lst_words[i:i + 1]) for i in range(0, len(lst_words), 1)]
        lst_corpus.append(lst_grams)
    return lst_corpus


'''
    Bag Of Words implementation
'''


def bagOfWords(text, ngram_range=(1, 1)):
    CountVec = CountVectorizer(ngram_range=ngram_range)  # to use bigrams ngram_range=(2,2)
    text_vec = CountVec.fit_transform(text)

    # return python dict instead of df
    # cv_dataframe = pd.DataFrame(Count_data.toarray(), columns=CountVec.get_feature_names_out())
    # res = list(map(lambda row: dict(zip(CountVec.get_feature_names_out(), row)), Count_data.toarray()))
    return text_vec


'''
   TFIDF implementation. corpus are all texts
'''


def TfIdf(text, ngram_range=(1, 1)):
    tfIdfVectorizer = TfidfVectorizer(ngram_range=ngram_range)
    vec_text = tfIdfVectorizer.fit_transform(text)

    # return python dict instead of df
    # cv_dataframe = pd.DataFrame(Count_data.toarray(), columns=CountVec.get_feature_names_out())
    # res = list(map(lambda row: dict(zip(tfIdfVectorizer.get_feature_names_out(), row)), Count_data.toarray()))
    return vec_text


'''
    Word2Vec implementation
    https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
'''


def word2Vec(texts: list):
    # CBOW (Continous Bag of Words):
    model1 = gensim.models.Word2Vec([texts], min_count=1, vector_size=300, window=7, sg=1)


'''
    Just testing the functions
'''


def main():
    # df = bagOfWords("Hallo, ich war mal groß und jetzt bin ich klein. Na und. Was machst du so?", ngram_range=(1,2))

    sentences = ['Hallo mein Name ist David', 'Das ist ein schöner Name', 'Du heißt Patrick, das ist ok']
    sentences_eng = ['Hello my name is Patrick', 'That is a beautiful name', 'Your name is David is not bad']
    df = TfIdf(sentences)
    print(df)

    word2Vec(sentences)

    pass


if __name__ == "__main__":
    main()
