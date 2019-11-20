import numpy as np
import math

class PretrainedEmbedder:
    def __init__(self, vocab, pretrained_path, num_dimensions=300):
        self.vocab = vocab
        self.pretrained_path = pretrained_path
        self.num_dimensions = num_dimensions
    
    def load_pretrained(self, filepath):
        print('Loading pretrained embedding data.........')
        pretrained = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                list_ = line.split(' ')
                word = list_[0]
                vect = np.array(list_[1:]).astype(np.float)
                self.num_dimensions = len(vect)
                pretrained[word] = vect
        print('Done loading pretrained')
        
        return pretrained

    def pretrained_embedder(self):
        print('Creating embedding weights.....')
        if self.pretrained_path is not None:
            pretrained = self.load_pretrained(self.pretrained_path)
            sd_rand = math.sqrt(3 / self.num_dimensions)
            embedding_weights = np.zeros((len(self.vocab), self.num_dimensions), dtype=float)
            for word in self.vocab:
                if word in pretrained.keys():
                    embedding_weights[int(self.vocab[word])] = pretrained[word]
                else:
                    embedding_weights[int(self.vocab[word])] = np.random.uniform(-sd_rand, sd_rand, self.num_dimensions)
        else:
            sd_rand = math.sqrt(3 / self.num_dimensions)
            embedding_weights = np.zeros((len(self.vocab), self.num_dimensions), dtype=float)
            for word in self.vocab:
                embedding_weights[int(self.vocab[word])] = np.random.uniform(-sd_rand, sd_rand, self.num_dimensions)
        
        print('Done creating embedding weights')
        print ('Num dimensions is %d' % self.num_dimensions)
        
        return (embedding_weights, self.num_dimensions)