import numpy as np 

class Vectorizer:
    def __init__(self, max_sentence_len: int, max_word_len: int, vocab: dict, char_vocab: dict, words: list):
        self.max_sentence_len = max_sentence_len
        self.max_word_len = max_word_len
        self.vocab = vocab
        self.char_vocab = char_vocab
        self.words = words
        # self.word_ignore_case = word_ignore_case
        # self.char_ignore_case = char_ignore_case

        self.word_emb_input = [[0] * self.max_sentence_len for _ in range(len(words))]
        self.char_emb_input = [[[0] * self.max_word_len for _ in range(self.max_sentence_len)] for _ in range(len(words))]
    
    def vectorizer(self):
        for sentence_index, sentence in enumerate(self.words):
            for word_index, word in enumerate(sentence):
                if word in self.vocab:
                    self.word_emb_input[sentence_index][word_index] = self.vocab[word]
                else:
                    self.word_emb_input[sentence_index][word_index] = self.vocab['_OOV_']
                if word != '<NUM>' and word != '<PUNCT>':
                    for char_index, char in enumerate(word):
                        if char_index >= self.max_word_len:
                            break 
                        if char in self.char_vocab:
                            self.char_emb_input[sentence_index][word_index][char_index] = self.char_vocab[char]
                        else:
                            self.char_emb_input[sentence_index][word_index][char_index] = self.char_vocab['_OOV_']
        
        return [np.asarray(self.word_emb_input), np.asarray(self.char_emb_input)]
                        
                