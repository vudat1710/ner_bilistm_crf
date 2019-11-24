from models.embedding_layer import EmbeddingLayer
from models.label_encoder import LabelEncoderModel
from keras.layers import Bidirectional, LSTM, Dense, Dropout, TimeDistributed
from keras_contrib.layers import CRF
from keras.models import Model
from keras.optimizers import Adam

class ModelTraining:
    def __init__(self, dropout: float, lr: float,n_classes: int, vocab_size: int, char_vocab_size: int, max_word_len: int, pos_len: int, chunk_len: int, word_emb_dimensions=300, char_emb_dimensions=100, word_hidden_dim=150, char_hidden_dim=100, word_emb_weights=None, char_emb_weights=None, pos_emb_weights=None, chunk_emb_weights=None, word_emb_trainable=False, char_emb_trainable=True, word_mask_zero=True, char_mask_zero=True):
        embedding = EmbeddingLayer(vocab_size, char_vocab_size, max_word_len, pos_len, chunk_len, word_emb_dimensions, char_emb_dimensions, char_hidden_dim, word_emb_weights, char_emb_weights, pos_emb_weights, chunk_emb_weights, word_emb_trainable, char_emb_trainable, word_mask_zero, char_mask_zero)
        self.inputs, self.emb_layer = embedding.embedding()
        self.word_hidden_dim = word_hidden_dim
        self.dropout = dropout
        self.n_classes = n_classes
        self.crf = CRF(self.n_classes, sparse_target=False)
        self.lr = lr

    def model_build(self):
        model = Bidirectional(LSTM(units=self.word_hidden_dim, return_sequences=True))
        self.emb_layer = Dropout(self.dropout)(self.emb_layer)
        lstm_after = model(self.emb_layer)
        lstm_after = Dropout(self.dropout)(lstm_after)
        # lstm_after = TimeDistributed(Dense(50, activation='relu'))(lstm_after)
        out = self.crf(lstm_after)

        training_model = Model(inputs=self.inputs, outputs=out, name="training_model")
        opt = Adam(self.lr)

        training_model.compile(loss=self.crf.loss_function, optimizer=opt, metrics=[self.crf.accuracy])

        return training_model
