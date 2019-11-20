import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

class LabelEncoderModel:
    def __init__(self, list_to_convert:list, max_sentence_len:int):
        self.list_to_convert = sum(list_to_convert, [])
        self.max_sentence_len = max_sentence_len
        self.embedding_weights = np.zeros((len(set(self.list_to_convert)), len(set(self.list_to_convert))), dtype=float)
        self.onehot_encoder = OneHotEncoder()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.list_to_convert)
        encoded_label = self.label_encoder.transform(self.list_to_convert)
        self.onehot_encoder.fit(encoded_label.reshape(-1,1))
    
    def onehot_vectorizer(self, target_list: list):
        converted_vec = [[0] * self.max_sentence_len for _ in range(len(target_list))]
        for sentence_index in range(len(target_list)):
            for index in range(len(target_list[sentence_index])):      
                ele = [target_list[sentence_index][index]]
                converted_vec[sentence_index][index] = self.label_encoder.transform(ele)[0]
        
        return converted_vec
    
    def get_emb_weights(self):
        print("----------Creating weights labels----------------")
        for c in self.label_encoder.classes_:
            ele = self.label_encoder.transform([c])
            self.embedding_weights[ele[0]] = self.onehot_encoder.transform(ele.reshape(-1,1)).toarray()[0]
        
        print("---------------------------------------------------")
        return [self.embedding_weights]