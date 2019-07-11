from NER_model import Ner
from NER_data import Data_set

LABELS_CATEGORY = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', "B-ORG", "I-ORG"]
TRAIN_DATA_PATH = "./data/example.train"
VOCAB_PATH = "./model/vocab.pk"
EPOCHES = 2

if __name__ == "__main__":
    data = Data_set(TRAIN_DATA_PATH,LABELS_CATEGORY)
    vocab = data.save_vocab(VOCAB_PATH)
    sentence,sen_tags= data.generate_data(vocab,200)

    ner = Ner(vocab,LABELS_CATEGORY)
    ner.train(sentence,sen_tags,EPOCHES)
