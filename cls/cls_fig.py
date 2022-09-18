# -*- coding: utf8 -*-
import json, numpy, collections, jieba.analyse
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
# https://www.sbert.net/docs/quickstart.html
# CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python context_cls.py

# INITb
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
new_list, test_list, data_tfidf, data_emb = [], [], [], []
dict_path, data_path, data_t_path = "", "", ""
train_len = 0

# PATH SETTING
num_tfidf = 1
num_clusters = 20
clu_type = "title"      # context, keyword
dataset_type = "drcd"   # cmrc

if clu_type=="keyword": save_path = "_output_master/" + dataset_type + "/" + clu_type + str(num_tfidf) + "/"
else: save_path = "_output_master/" + dataset_type + "/" + clu_type + "/"

if dataset_type=="drcd":
    dict_path = "jieba/jieba-zh_TW-master/jieba/dict.txt"
    data_path = "../DRCD/DRCD_training.json"
    data_t_path = "../DRCD/DRCD_test.json"
    data_new_path = save_path + "DRCD_training.json"
    data_new_t_path =  save_path + "DRCD_test.json"
else:
    dict_path = "jieba/dict.txt.big"
    data_path = "../cmrc2018/train.json"
    data_t_path = "../cmrc2018/dev.json"
    data_new_path = save_path + "train.json"
    data_new_t_path = save_path + "dev.json"

cnt_path = save_path + "count.txt"
fig_path = save_path + "fig.png"

# LOAD DATA
print("== LOAD DATA ==") 
with open(data_path, encoding = 'utf-8') as data_json, open(data_t_path, encoding = 'utf-8') as data_t_json:
    dataset = collections.OrderedDict(json.load(data_json))
    dataset_t = collections.OrderedDict(json.load(data_t_json))
new_data = dataset
new_data_t = dataset_t
        
# if clu_type=="title":
#     new_list = [[article['title']] for article in dataset['data']]
#     test_list = [[article['title']] for article in dataset_t['data']]
# elif clu_type=="context" or clu_type=="keyword":
#     new_list = [[paragraph["context"] for paragraph in article['paragraphs']] for article in dataset['data']]
#     test_list = [[paragraph["context"] for paragraph in article['paragraphs']] for article in dataset_t['data']]
# else: print("clu_type error")


if clu_type=="title":
    for article in dataset['data']:
        if new_list == []: new_list = [article["title"]]
        else: new_list.append(article["title"])
    for article in dataset_t['data']:
        if test_list == []: test_list = [article["title"]]
        else: test_list.append(article["title"])
elif clu_type=="context" or clu_type=="keyword":
    for article in dataset['data']:
        for paragraph in article['paragraphs']:
            if new_list == []: new_list = [paragraph["context"]]
            else: new_list.append(paragraph["context"])
    for article in dataset_t['data']:
        for paragraph in article['paragraphs']:
            if test_list == []: test_list = [paragraph["context"]]
            else: test_list.append(paragraph["context"])


print("new_list len: ", len(new_list), "\ntest_list len: ", len(test_list)) 

# print(arr_emb.shape)    # context (8014, 384)   title (1960, 384)     title(+test) (2338, 384)
train_len = len(new_list)
new_list = list(new_list+test_list)

# RUN TFIDF
print("== RUN TFIDF ==") 
if clu_type=="keyword":
    jieba.set_dictionary(dict_path)
    for paragraph in new_list:
        sentence = str()
        if isinstance(paragraph, str):
            words = jieba.analyse.extract_tags(paragraph, topK=num_tfidf)
            for word in words:
                if sentence == "": sentence = str(word)
                else: sentence = sentence + "，" + str(word)
            data_tfidf.append(sentence)

new_list = data_tfidf
print("new_list len: ", len(new_list), "\ntest_list len: ", len(test_list)) 

print(" === CLUSTERING ===")
arr_emb = model.encode(new_list)
arr_emb = numpy.array(arr_emb)
cluster = KMeans(n_clusters=num_clusters).fit(arr_emb)              # k-means
# cluster = DBSCAN(eps=3,min_samples=10).fit(arr_emb)               # dbscan

# 2d
L_sk = PCA(2).fit_transform(arr_emb)
plt.scatter(L_sk[:,0],L_sk[:,1],c=cluster.labels_, cmap='Spectral') 
plt.savefig(fig_path)

label_list = list(cluster.labels_)
new_id_list = []    
for i in label_list:
    i = int(i)
    new_id_list.append(i)

print(" === OUTPUTING ===")
cnt_list = collections.OrderedDict()
for i in new_id_list:
    if i not in cnt_list: cnt_list[i] = 1
    else: cnt_list[i] = cnt_list[i] + 1      

with open(cnt_path, "w", encoding='utf8') as cnt_file, open(data_new_path, "w", encoding='utf8') as n_data_file, open(data_new_t_path, "w", encoding='utf8') as n_data_t_file:
    json.dump(cnt_list, cnt_file, ensure_ascii=False)
    json.dump(new_data, n_data_file, ensure_ascii=False)
    json.dump(new_data_t, n_data_t_file, ensure_ascii=False)