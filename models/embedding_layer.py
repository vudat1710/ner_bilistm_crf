import numpy as np 
import keras.backend as K
from keras.utils import to_categorical
from keras.layers import Input, Dropout, Lambda, LSTM, Embedding, Bidirectional, TimeDistributed, Concatenate

class EmbeddingLayer:
    def __init__(self, vocab_size: int, char_vocab_size: int, max_word_len: int, pos_len: int, chunk_len: int, word_emb_dimensions=300, char_emb_dimensions=100, char_hidden_dim=100, word_emb_weights=None, char_emb_weights=None, pos_emb_weights=None, chunk_emb_weights=None, word_emb_trainable=False, char_emb_trainable=True, word_mask_zero=True, char_mask_zero=True):
        self.vocab_size = vocab_size
        self.char_vocab_size = char_vocab_size
        self.max_word_len = max_word_len
        self.word_emb_dimensions = word_emb_dimensions
        self.char_emb_dimensions = char_emb_dimensions
        self.char_hidden_dim = char_hidden_dim
        self.word_emb_weights = word_emb_weights
        self.char_emb_weights = char_emb_weights
        self.word_emb_trainable = word_emb_trainable
        self.char_emb_trainable = char_emb_trainable
        self.word_mask_zero = word_mask_zero
        self.char_mask_zero = char_mask_zero
        self.pos_len = pos_len
        self.chunk_len = chunk_len
        self.pos_emb_weights = pos_emb_weights
        self.chunk_emb_weights = chunk_emb_weights

    def embedding(self):
        if self.word_emb_weights is not None:
            self.word_emb_weights = [self.word_emb_weights]
        
        if self.char_emb_weights is not None:
            self.char_emb_weights = [self.char_emb_weights]
        
        word_input_layer = Input(shape=(None,), name="Input_word")
        char_input_layer = Input(shape=(None, None,), name="Input_char")
        pos_input_layer = Input(shape=(None,), name="Input_pos")
        chunk_input_layer = Input(shape=(None,), name="Input_chunk")
        
        word_emb_layer = Embedding(input_dim=self.vocab_size,
                        output_dim=self.word_emb_dimensions,
                        mask_zero=self.word_mask_zero,
                        weights=self.word_emb_weights,
                        trainable=self.word_emb_trainable,
                        name="Embedding_word")(word_input_layer)
        char_emb_layer = Embedding(input_dim=self.char_vocab_size,
                        output_dim=self.char_emb_dimensions,
                        mask_zero=self.char_mask_zero,
                        weights=self.char_emb_weights,
                        trainable=self.char_emb_trainable,
                        name="Embedding_Char_Pre")(char_input_layer)
        pos_emb_layer = Embedding(input_dim=self.pos_len,
                        output_dim=self.pos_len,
                        weights=self.pos_emb_weights,
                        trainable=False,
                        name="Embedding_pos")(pos_input_layer)
        chunk_emb_layer = Embedding(input_dim=self.chunk_len,
                        output_dim=self.chunk_len,
                        weights=self.chunk_emb_weights,
                        trainable=False,
                        name="Embedding_chunk")(chunk_input_layer)  

        s = K.shape(char_emb_layer)
        char_emb_layer = Lambda(lambda x: K.reshape(x, shape=(-1, s[-2], self.char_emb_dimensions)))(char_emb_layer)   
        fwd_state = LSTM(self.char_hidden_dim, return_state=True, name='fw_char_lstm')(char_emb_layer)[-2]
        bwd_state = LSTM(self.char_hidden_dim, return_state=True, go_backwards=True, name='bw_char_lstm')(char_emb_layer)[-2]
        char_emb_layer = Concatenate(axis=-1)([fwd_state, bwd_state])
        char_emb_layer = Lambda(lambda x: K.reshape(x, shape=[-1, s[1], 2 * self.char_hidden_dim]))(char_emb_layer)
        # char_hidden_layer = Bidirectional(
        #     LSTM(
        #         units=self.char_hidden_dim,
        #         input_shape=(self.max_word_len, self.char_vocab_size),
        #         return_sequences=False,
        #         return_state=True
        #     ), name="BiLSTM_Char"
        # )
        # if not isinstance(char_hidden_layer, list):
        #     char_hidden_layer = [char_hidden_layer]
        # for i, layer in enumerate(char_hidden_layer):
        #     if i == len(char_hidden_layer) - 1:
        #         name = "Embedding_Char"
        #     else:
        #         name = "Embedding_Char_Pre%d" % (i+1)
        #     char_emb_layer = TimeDistributed(layer=layer, name=name)(char_emb_layer)
        emb_layer = Concatenate(axis=-1,name="Embedding")([char_emb_layer, pos_emb_layer, chunk_emb_layer, word_emb_layer, ])
        # emb_layer = Concatenate(axis=-1,name="Embedding")([char_emb_layer, word_emb_layer])

        return [char_input_layer, word_input_layer, pos_input_layer, chunk_input_layer], emb_layer
