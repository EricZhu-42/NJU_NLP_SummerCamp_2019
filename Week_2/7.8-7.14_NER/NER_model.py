import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras import Sequential
from keras_contrib.layers import CRF
from keras.layers import Embedding ,Bidirectional,LSTM
from keras.models import load_model

BATCH_SIZE = 32
MODEL_PATH = "./model/crf.h5"

class Ner:
    def __init__(self,vocab,labels_category,Embedding_dim=200):
        self.Embedding_dim = Embedding_dim
        self.vocab = vocab
        self.labels_category = labels_category
        self.model = self.build_model()
        try:
            self.model.load_weights(MODEL_PATH)
            print('Loading model from {:s}'.format(MODEL_PATH))
        except Exception as e:
            print(e, "creating new model.")

    def build_model(self):
        model = Sequential()
        model.add(Embedding(len(self.vocab),self.Embedding_dim,mask_zero=True))  # Random embedding
        model.add(Bidirectional(LSTM(100, return_sequences=True)))
        crf = CRF(len(self.labels_category), sparse_target=True)
        model.add(crf)
        model.summary()
        model.compile('adam', loss=crf.loss_function, metrics=[crf.accuracy])
        return model

    # 训练后保存模型
    def train(self,data,label,EPOCHS):
        self.model.fit(data,label,batch_size=BATCH_SIZE,epochs=EPOCHS)
        self.model.save(MODEL_PATH)

    def predict(self,data,maxlen):
        char2id = [self.vocab.get(i) for i in data]
        #pad_num = maxlen - len(char2id)
        input_data = pad_sequences([char2id],maxlen)
        result = self.model.predict(input_data)[0][-len(data):]
        result_label = [np.argmax(i) for i in result]
        return result_label