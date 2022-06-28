from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K
from keras.models import Sequential, Model


def createConvolutionalModel(vocabular_size, embedding_dim: int = 50, max_length: int = 100):
    model = Sequential()
    model.add(layers.Embedding(vocabular_size, embedding_dim, input_length=max_length))
    model.add(layers.Conv1D(128, 5, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def basicDeepLearningModel(vocabular_size, embedding_dim: int = 50, max_length: int = 100):
    model = Sequential()

    model.add(layers.Embedding(input_dim=vocabular_size,
                               output_dim=embedding_dim,
                               input_length=max_length))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def deepLearningModelWithEmbeddingMatrix(vocabular_size, embedding_matrix, embedding_dim: int = 50, max_length: int = 100):
    model = Sequential()
    model.add(layers.Embedding(vocabular_size, embedding_dim,
                               weights=[embedding_matrix],
                               input_length=max_length,
                               trainable=False))
    model.add(layers.GlobalMaxPool1D())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()