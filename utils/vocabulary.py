import random

class Vocabulary:
    def __init__(self, words:list):
        self.vocab = {}
        self.char_vocab = {}

        self.vocab['_PAD_'] = 0
        self.vocab['_OOV_'] = 1
        self.char_vocab['_PAD_'] = 0
        self.char_vocab['_OOV_'] = 1

        for sent in words:
            for word in sent:
                self.add_word(word)
                self.add_char(word)
        
        print('-------------------------------')
        print('Vocabulary with %d words created' % len(self.vocab))
        print('Sample words from vocab: ', dict(random.sample(self.vocab.items(), 50)))
        print('-------------------------------\n\n\n')

        print('-------------------------------')
        print('Vocabulary with %d characters created' % len(self.char_vocab))
        print('Sample characters from vocab: ', dict(random.sample(self.char_vocab.items(), 50)))
        print('-------------------------------')
    
    def add_word(self, word:str):
        if word not in self.vocab.keys():
            self.vocab[word] = len(self.vocab)
    
    def add_char(self, word:str):
        if word != '<NUM>' and word != '<PUNCT>':
            for char in word:
                if char not in self.char_vocab.keys():
                    self.char_vocab[char] = len(self.char_vocab)
    
    def get_word_vocab(self):
        return self.vocab
    
    def get_char_vocab(self):
        return self.char_vocab