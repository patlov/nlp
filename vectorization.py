from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import pandas as pd



'''
    Bag Of Words implementation
'''
def bagOfWords(text: str, ngram_range=(1, 1)):
    CountVec = CountVectorizer(ngram_range=ngram_range)  # to use bigrams ngram_range=(2,2)
    Count_data = CountVec.fit_transform([text])
    cv_dataframe = pd.DataFrame(Count_data.toarray(), columns=CountVec.get_feature_names())
    return cv_dataframe


'''
    Bag Of Words implementation. corpus are all texts
'''
def TfIdf(corpus: list, ngram_range=(1, 1)):
    vectorizer = TfidfVectorizer(analyzer="word", norm="l2", ngram_range=ngram_range)

    matrix = vectorizer.fit_transform(corpus)
    return matrix



'''
    Just testing the functions
'''
def main():

    # df = bagOfWords("Hallo, ich war mal groß und jetzt bin ich klein. Na und. Was machst du so?", ngram_range=(1,2))

    sentences = ['Hallo mein Name ist David', 'Das ist ein schöner Name', 'Du heißt Patrick, das ist kein schöner Name']
    df = TfIdf(sentences)

    print(df)

    pass


if __name__ == "__main__":
    main()
