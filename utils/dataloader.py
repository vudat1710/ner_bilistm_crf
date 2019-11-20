import pandas as pd
from utils.utils import get_processing_word

class DataLoader:
    def __init__(self, filepath:str, lowercase=True):
        self.filepath = filepath
        self.words = []
        self.poss = []
        self.chunks = []
        self.labels = []
        self.max_sentence_len = 0
        self.max_word_len = 0
        self.get_processing_word = get_processing_word(lowercase=lowercase)

    def get_data(self):
        res = []
        with open(self.filepath, 'r') as f:
            _l = []
            for line in f.readlines():
                line = line.strip()
                if line != "":
                    _l.append(line.split('\t'))
                else:
                    res.append(_l)
                    _l = []
        f.close()

        return res
    
    def get_all_train_tokens(self):
        sentences = self.get_data()
        for sent in sentences:
            if len(sent) > self.max_sentence_len:
                self.max_sentence_len = len(sent)
            _words = []
            _poss = []
            _chunks = []
            _labels = []
            for token in sent:
                word = self.get_processing_word(token[0])
                _words.append(word)
                if len(token[0]) > self.max_word_len:
                    self.max_word_len = len(token[0])
                _poss.append(token[1])
                _chunks.append(token[2])
                _labels.append(token[3])

            self.words.append(_words)
            self.poss.append(_poss)
            self.chunks.append(_chunks)
            self.labels.append(_labels)
        
        return self.words, self.poss, self.chunks, self.labels
    
    def get_required_max_len(self):
        return self.max_sentence_len, self.max_word_len