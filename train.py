from utils.dataloader import DataLoader
from utils.pretrained_embedder import PretrainedEmbedder
from utils.vocabulary import Vocabulary
from utils.vectorizer import Vectorizer
from models.model import ModelTraining
from models.label_encoder import LabelEncoderModel

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
import os

class Trainer:
    def __init__(self, args):
        self.args = args
        train = DataLoader(self.args.trainpath)
        dev = DataLoader(self.args.devpath)

        self.train_words, self.train_poss, self.train_chunks, self.train_labels = train.get_all_train_tokens()
        self.train_max_sentence_len, self.train_max_word_len = train.get_required_max_len()
        self.dev_words, self.dev_poss, self.dev_chunks, self.dev_labels = dev.get_all_train_tokens()
        self.dev_max_sentence_len, self.dev_max_word_len = dev.get_required_max_len()  

        vocabulary = Vocabulary(self.train_words)     
        self.vocab = vocabulary.get_word_vocab()
        self.char_vocab = vocabulary.get_char_vocab()

        self.train_vect = Vectorizer(self.train_max_sentence_len, self.train_max_word_len, self.vocab, self.char_vocab, self.train_words)
        self.dev_vect = Vectorizer(self.dev_max_sentence_len, self.dev_max_word_len, self.vocab, self.char_vocab, self.dev_words)

        self.poss_vect = LabelEncoderModel(self.train_poss,self.train_max_sentence_len)
        self.chunks_vect = LabelEncoderModel(self.train_chunks,self.train_max_sentence_len)
        self.labels_vect = LabelEncoderModel(self.train_labels,self.train_max_sentence_len)

        #st wrong here
        self.pos_emb_weights = self.poss_vect.get_emb_weights()
        self.chunk_emb_weights = self.chunks_vect.get_emb_weights()
        self.word_emb_weights, self.word_emb_dimensions = PretrainedEmbedder(self.vocab, self.args.pretrained_path).pretrained_embedder()
        self.model = ModelTraining(self.args.dropout, self.args.lr, len(set(sum(self.train_labels, []))), len(self.vocab), len(self.char_vocab), self.train_max_word_len, 
                    len(set(sum(self.train_poss,[]))), len(set(sum(self.train_chunks,[]))), word_emb_dimensions=self.word_emb_dimensions, 
                    word_emb_weights=self.word_emb_weights, pos_emb_weights=self.pos_emb_weights, chunk_emb_weights=self.chunk_emb_weights).model_build()

    
    def helper(self):
        _l = []
        for filename in os.listdir(self.args.checkpoint_dir):
            if filename.endswith('.h5'):
                _l.append(int(filename[-5:-3]))
        if max(_l) >= 10:
            return str(max(_l))
        else:
            return str(' %d' % max(_l))
    
    def train(self):
        if not os.path.exists(self.args.checkpoint_dir):
            os.mkdir(self.args.checkpoint_dir)
        for filename in os.listdir(self.args.checkpoint_dir):
            os.remove(os.path.join(self.args.checkpoint_dir, filename))

        word_emb_input, char_emb_input = self.train_vect.vectorizer()
        poss_emb_input = self.poss_vect.onehot_vectorizer(self.train_poss)
        chunks_emb_input = self.chunks_vect.onehot_vectorizer(self.train_chunks)
        labels = self.labels_vect.onehot_vectorizer(self.train_labels)

        word_emb_input_dev, char_emb_input_dev = self.dev_vect.vectorizer()
        poss_emb_input_dev = self.poss_vect.onehot_vectorizer(self.dev_poss)
        chunks_emb_input_dev = self.chunks_vect.onehot_vectorizer(self.dev_chunks)
        labels_dev = self.labels_vect.onehot_vectorizer(self.dev_labels)

        self.model.fit(
            [char_emb_input, word_emb_input, poss_emb_input, chunks_emb_input],
            labels,
            epochs=self.args.epochs,
            batch_size=self.args.batch_size,
            validation_data=([char_emb_input_dev, word_emb_input_dev, poss_emb_input_dev, chunks_emb_input_dev], labels_dev),
            verbose=1
        )

        self.model.summary()

