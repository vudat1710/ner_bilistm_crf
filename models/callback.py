from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import numpy as np 
from keras.callbacks import Callback

def pred2label(pred, label_encoder):
    out = []
    for pred_i in pred:
        out_i = []
        for p in pred_i:
            p_i = np.argmax(p)
            out_i.append(label_encoder.inverse_transform([p_i])[0])
        out.append(out_i)
    return out

class AnSelCB(Callback):
    def __init__(self, y, inputs, train_y, train_inputs, label_encoder):
        self.val_y = y
        self.val_inputs = inputs
        self.train_y = train_y
        self.train_inputs = train_inputs
        self.label_encoder = label_encoder
    
    def on_epoch_end(self, epoch, logs={}):
        pred = self.model.predict(self.val_inputs)
        train_pred = self.model.predict(self.train_inputs)
        pred_labels = pred2label(pred, self.label_encoder)
        train_pred_labels = pred2label(train_pred, self.label_encoder)
        f1 = f1_score(self.val_y, pred_labels)
        print("F1-score: {:.1%}".format(f1_score(self.val_y, pred_labels)))
        print("F1-score: {:.1%}".format(f1_score(self.train_y, train_pred_labels)))
        logs['f1'] = f1