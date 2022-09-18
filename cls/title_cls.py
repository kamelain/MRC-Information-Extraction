# -*- coding: utf8 -*-
import json, numpy, collections
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
# https://www.sbert.net/docs/quickstart.html

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
data_path = "../DRCD/DRCD_training.json"
emb_path = "emb2.json"
data_emb = []

# iris=datasets.load_iris()
# X=iris.data
# print(type(X))    <class 'numpy.ndarray'>
# X=X[:,2:4]
# print(type(X))    <class 'numpy.ndarray'>
# print(X)          [[],[],...]
# print(X.shape)    (150, 2)

with open(data_path, encoding = 'utf-8') as data_json:
    dataset = collections.OrderedDict(json.load(data_json))

for article in dataset['data']:
    for paragraph in article['paragraphs']:
        temp = {}
        temp['id'] = paragraph['id']
        temp['context'] = paragraph['context']
        temp['emb'] = model.encode(paragraph['context'])
        data_emb.append(temp)
        # print(temp['emb'].shape)
        
# array = data_emb['emb']
# print(type(array))
# clustering=DBSCAN(eps=0.3,min_samples=10).fit(array)
# print(clustering.shape)
# print(clustering.labels_)

list_emb = [i['emb'] for i in data_emb]
arr_emb = numpy.array(list_emb)
# print(arr_emb.shape)    (8014, 384)

clustering=DBSCAN(eps=3,min_samples=10).fit(arr_emb)
# (0.3,10)(0.5,5) all -1   (1.5,10) almost -1      
# print(clustering.labels_)   [-1 -1 -1 ... -1 -1 -1]
plt.scatter(arr_emb[:,0],arr_emb[:,1],c=clustering.labels_)  
plt.show()
label_list = list(clustering.labels_)
new_list = []    
for i in label_list:
    i = int(i)
    new_list.append(i)

with open(emb_path, "w", encoding='utf8') as emb_file:
    json.dump(new_list, emb_file, ensure_ascii=False)

# print(len(data_emb))