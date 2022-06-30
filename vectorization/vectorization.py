from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
nltk.download('punkt')





def bagOfWords(text: str, ngram_range=(1, 1)):
    """
        Bag Of Words implementation
    """
    CountVec = CountVectorizer(ngram_range=ngram_range)  # to use bigrams ngram_range=(2,2)
    Count_data = CountVec.fit_transform([text])

    # return python dict instead of df
    # cv_dataframe = pd.DataFrame(Count_data.toarray(), columns=CountVec.get_feature_names_out())
    res = list(map(lambda row: dict(zip(CountVec.get_feature_names_out(), row)), Count_data.toarray()))
    return res[0]


def TfIdf(text: str, ngram_range=(1, 1), min_df: int = 3):
    """
       TFIDF implementation. corpus are all texts
    """
    tfIdfVectorizer = TfidfVectorizer(ngram_range=ngram_range)
    Count_data = tfIdfVectorizer.fit_transform([text])

    # return python dict instead of df
    # cv_dataframe = pd.DataFrame(Count_data.toarray(), columns=CountVec.get_feature_names_out())
    res = list(map(lambda row: dict(zip(tfIdfVectorizer.get_feature_names_out(), row)), Count_data.toarray()))
    return res[0]


def getVectorizer(vecType, ngram_range=(1, 3), max_df=0.3, min_df=7):
    """
        returns the vectorizer depending on the vectorization type
    """
    if vecType.value == 2:  # bag of words
        return CountVectorizer(ngram_range=ngram_range, max_df=max_df, min_df=min_df)
    elif vecType.value == 4:  # tfidf
        return TfidfVectorizer(ngram_range=ngram_range, norm="l2", max_df=max_df, min_df=min_df)


