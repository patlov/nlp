from typing import List

from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences


def createPaddingSequence(x_train, x_test, maxlen: int = 100):
    tokenizer = Tokenizer(lower=True, split=' ', oov_token="NaN",
                          filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
    tokenizer.fit_on_texts(x_train)

    x_train_seq = tokenizer.texts_to_sequences(x_train)
    x_test_seq = tokenizer.texts_to_sequences(x_test)
    voc_size = len(tokenizer.word_index) + 1

    X_train = pad_sequences(x_train_seq, maxlen=maxlen, padding="post", truncating="post")
    X_test = pad_sequences(x_test_seq, maxlen=maxlen, padding="post", truncating="post")

    return [X_train, X_test, voc_size]

