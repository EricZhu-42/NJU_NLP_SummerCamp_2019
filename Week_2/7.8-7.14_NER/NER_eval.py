from NER_data import Data_set
from NER_model import MODEL_PATH, Ner
from NER_train import LABELS_CATEGORY, VOCAB_PATH
import pickle

TEST_DATA_PATH = "./data/example.test"
with open(VOCAB_PATH,'rb') as f:
    vocab = pickle.load(f)

ner = Ner(vocab, LABELS_CATEGORY)
data = Data_set(TEST_DATA_PATH, LABELS_CATEGORY).processing_data()
data = data[:1000]
"""
FORMAT:

[
    [['你','O'],['好','O'],['！','O']],
    [['南','B-ORG'],['京'，'I-ORG'],['大','I-ORG'],['学','I-ORG']]
]
"""
sentence_list = list()
true_label_list = list()
pre_label_list = list()
counter = 0

for sentence in data:
    counter += 1
    if counter%20==0:
        print("{:d} of {:d}.".format(counter,len(data)))
    sentence_str = str()
    true_label = list()
    for ses in sentence:
        sentence_str += ses[0]
        true_label.append(ses[1])
    try:
        res = ner.predict(sentence_str,200)
    except Exception:
        continue
    res2label = [LABELS_CATEGORY[i] for i in res]
    pre_label_list += res2label
    true_label_list += true_label

len_data = len(pre_label_list)
count_category = [0 for i in LABELS_CATEGORY]
precison_list = [0 for i in LABELS_CATEGORY]
recall_list = [0 for i in LABELS_CATEGORY]
F1_list = [0 for i in LABELS_CATEGORY]
zip_dict = {
    'O':0,
    'B-PER':1,
    'I-PER':2,
    'B-LOC':3,
    'I-LOC':4,
    "B-ORG":5,
    "I-ORG":6
}
true_label_list = list(map(lambda x:zip_dict[x],true_label_list))
pre_label_list = list(map(lambda x:zip_dict[x],pre_label_list))
print('-----------------RESULTS:---------------------')
for i in range(len(LABELS_CATEGORY)):
    count_category[i] = true_label_list.count(i)
    TP = len([j for j in range(len_data) if true_label_list[j]==i and pre_label_list[j]==i ])+1
    FP = len([j for j in range(len_data) if true_label_list[j]!=i and pre_label_list[j]==i ])+1
    FN = len([j for j in range(len_data) if true_label_list[j]==i and pre_label_list[j]!=i ])+1
    precision = 1.0*TP/(TP+FP)
    precison_list[i] = precision
    recall = 1.0*TP/(TP+FN)
    recall_list[i] = recall
    F1 = 2.0*precision*recall/(precision+recall)
    F1_list[i] = F1
    print('{:s}, precision={:.4f}, recall={:.4f}, F1={:.4f}.'.format(LABELS_CATEGORY[i],precision,recall,F1))
precision_mean = sum([count_category[i]*precison_list[i] for i in range(len(LABELS_CATEGORY))])/len_data
recall_mean = sum([count_category[i]*recall_list[i] for i in range(len(LABELS_CATEGORY))])/len_data
F1_mean = sum([count_category[i]*F1_list[i] for i in range(len(LABELS_CATEGORY))])/len_data
print('Mean_value, precision={:.4f}, recall={:.4f}, F1={:.4f}.'.format(precision_mean,recall_mean,F1_mean))