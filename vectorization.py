from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd
import gensim
import nltk
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()

nltk.download('stopwords')
nltk.download('punkt')
'''
    Bag Of Words implementation
'''


def bagOfWords(text: str, ngram_range=(1, 1)):
    CountVec = CountVectorizer(ngram_range=ngram_range)  # to use bigrams ngram_range=(2,2)
    Count_data = CountVec.fit_transform([text])
    cv_dataframe = pd.DataFrame(Count_data.toarray(), columns=CountVec.get_feature_names())
    return cv_dataframe


'''
    Word2Vec implementation
    https://www.geeksforgeeks.org/python-word-embedding-using-word2vec/
'''


def word2Vec(texts: list):
    # CBOW (Continous Bag of Words):
    model1 = gensim.models.Word2Vec([texts], min_count=1, vector_size=100, window=5)
    print("Cosine similarity for BOW (Continous Bag of Words) of sentences: " + texts[0] + ' and ' +
          texts[1] + ' is: ' + str(model1.wv.similarity(texts[0], texts[1])))

    model2 = gensim.models.Word2Vec([texts], min_count=1, vector_size=100, window=5, sg=1)
    # Skip Gram
    print("Cosine similarity for skip gram of sentences: " + texts[0] + ' and ' + texts[1] + ' is: ' +
          str(model2.wv.similarity(texts[0], texts[1])))


'''
    tokenization of words and stopwords removal
'''


def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return filtered_text


'''
   Word stemming based on porter stemmer e.g. science -> scienc, uses -> use
'''


def stem_words(text):
    word_tokens = word_tokenize(text)
    stems = [stemmer.stem(word) for word in word_tokens]
    return stems

'''
    Lemmatization of text e.g. uses -> use, processes -> process
'''

def lemmatize_word(text):
    word_tokens = word_tokenize(text)
    lemmas = [lemmatizer.lemmatize(word, pos='v') for word in word_tokens]
    return lemmas


'''
    Bag Of Words implementation. corpus are all texts
'''


def TfIdf(corpus: list, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(analyzer="word", norm="l2", ngram_range=ngram_range)
    matrix = vectorizer.fit_transform(corpus)
    tfidf_df = pd.DataFrame(matrix.toarray(), columns=vectorizer.get_feature_names())
    return tfidf_df


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
    tokenizedSentences = [remove_stopwords(sentence) for sentence in sentences_eng]
    print(tokenizedSentences)

    pass


if __name__ == "__main__":
    main()
