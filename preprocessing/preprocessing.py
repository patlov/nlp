
# first download stopwords
import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords
from germalemma import GermaLemma
import pos_tagging
import pickle
import os

POS_TAGGING_GERMAN_PICKLE = 'nltk_german_classifier_data.pickle'

def removeStopwords(comment):
    stops = set(stopwords.words("german"))
    cleaned_comment = []
    for word in comment.split():
        if word not in stops:
            cleaned_comment.append(word)
    return " ".join(cleaned_comment)


def lemmatizeSentence(comment):
    lemmatizer = GermaLemma()
    tokens = nltk.word_tokenize(comment)

    if not os.path.exists(POS_TAGGING_GERMAN_PICKLE):
        tagger = pos_tagging.trainAndSave_POSModel()
    else:
        with open(POS_TAGGING_GERMAN_PICKLE, 'rb') as f:
            tagger = pickle.load(f)

    tokens_with_pos = pos_tagging.POSTaggingWithTagger(tagger, tokens)

    words = []
    for i, token_with_pos in enumerate(tokens_with_pos):
        try:
            lemmatized = lemmatizer.find_lemma(token_with_pos[0], token_with_pos[1])
        except ValueError: # "ich" or other prepositions are not found
            words.append(token_with_pos[0]) # add the original word
            continue
        words.append(lemmatized)

    return ' '.join(words)


def main():

    lem_tok = lemmatizeSentence("Hallo, ich war mal größer")
    print(lem_tok)
    pass

if __name__ == "__main__":
        main()

