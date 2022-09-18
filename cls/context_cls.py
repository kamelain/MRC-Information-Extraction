# -*- coding: utf8 -*-
import json, numpy, collections, jieba.analyse
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
# https://www.sbert.net/docs/quickstart.html
# CUDA_VISIBLE_DEVICES=0 PYTHONIOENCODING=utf-8 python context_cls.py

data_check_path = "dataset_ext/DRCD/check_ori.json"
dict_path = "jieba/dict.txt.big"
# dict_path = "jieba/jieba-zh_TW-master/jieba/dict.txt"
tfidf_numb = 1

data_path = "../DRCD/DRCD_training.json"
data_t_path = "../DRCD/DRCD_test.json"
# data_new_path = "dataset_cls/DRCD/DRCD_training.json"
# data_new_t_path = "dataset_cls/DRCD/DRCD_test.json"
data_new_path = "dataset_ext/DRCD/DRCD_training3.json"
data_new_t_path = "dataset_ext/DRCD/DRCD_test3.json"
emb_path = "emb.json"
cnt_path = "count.txt"
fig_path = "fig.png"
# data_path = "../cmrc2018/train.json"
# data_t_path = "../cmrc2018/dev.json"
# # data_new_path = "dataset_cls/cmrc2018/train.json"
# # data_new_t_path = "dataset_cls/cmrc2018/dev.json"
# data_new_path = "dataset_ext/cmrc2018/train2.json"
# data_new_t_path = "dataset_ext/cmrc2018/dev2.json"
# emb_path = "emb_c.json"
# cnt_path = "count_c.txt"
# fig_path = "fig_c.png"

# @1 768, 5105, 75: ~436
# {"0": 5105, "1": 44, "2": 96, "3": 10, "4": 62, "5": 32, "6": 47, "7": 436, "8": 84, "9": 36, "10": 43, "11": 22, "12": 44, "-1": 768, "13": 35, "14": 26, "15": 45, "16": 77, "17": 35, "18": 43, "19": 151, "20": 55, "21": 17, "22": 68, "23": 22, "24": 59, "25": 47, "26": 11, "27": 36, "28": 22, "29": 38, "30": 33, "42": 40, "64": 17, "31": 51, "32": 36, "33": 39, "34": 22, "35": 10, "36": 32, "37": 19, "44": 36, "38": 34, "39": 229, "40": 43, "41": 30, "43": 14, "45": 13, "46": 11, "47": 68, "48": 13, "49": 12, "69": 10, "74": 10, "50": 96, "51": 20, "52": 51, "53": 23, "54": 36, "55": 72, "56": 28, "71": 12, "57": 11, "58": 16, "59": 28, "60": 13, "61": 12, "66": 15, "62": 17, "63": 18, "67": 11, "65": 16, "72": 10, "68": 22, "70": 18, "73": 21, "75": 10}
# @2 346, 8483, 7
# {"0": 8483, "-1": 346, "1": 10, "2": 40, "3": 15, "4": 25, "5": 39, "6": 20, "7": 36}
# @3 307, 8565, 4
# {"0": 8565, "-1": 307, "1": 77, "2": 16, "3": 38, "4": 11}


model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
data_emb = []
train_len = 0

with open(data_path, encoding = 'utf-8') as data_json, open(data_t_path, encoding = 'utf-8') as data_t_json:
    dataset = collections.OrderedDict(json.load(data_json))
    dataset_t = collections.OrderedDict(json.load(data_t_json))
new_data = dataset
new_data_t = dataset_t
        
# title as embedding
# for article in dataset['data']:
#     temp = {}
#     temp['id'] = article['id']
#     temp['context'] = article['title']
#     temp['emb'] = model.encode(article['title'])
#     data_emb.append(temp)

# for article in dataset_t['data']:
#     temp = {}
#     temp['id'] = article['id']
#     temp['context'] = article['title']
#     temp['emb'] = model.encode(article['title'])
#     data_emb.append(temp)

# train_len = len(data_emb)

# list_emb = [i['emb'] for i in data_emb]
# arr_emb = numpy.array(list_emb)
# print(arr_emb.shape)    # context (8014, 384)   title (1960, 384)     title(+test) (2338, 384)

# ext context
data_tfidf_strl = data_tfidf = new_list = test_list = []

for article in dataset['data']:
    for paragraph in article['paragraphs']:
        if new_list == []:
            new_list = [paragraph["context"]]
        else:
            new_list.append(paragraph["context"])

for article in dataset_t['data']:
    for paragraph in article['paragraphs']:
        if test_list == []:
            test_list = [paragraph["context"]]
        else:
            test_list.append(paragraph["context"])

# new_list = [[paragraph["context"] for paragraph in article['paragraphs']] for article in dataset['data']]
# test_list = [[paragraph["context"] for paragraph in article['paragraphs']] for article in dataset_t['data']]
train_len = len(new_list)
new_list = new_list + test_list
new_list = list(new_list)

jieba.set_dictionary(dict_path)
for paragraph in new_list:
    sentence = str()

    if isinstance(paragraph, str):
        words = jieba.analyse.extract_tags(paragraph, topK=tfidf_numb)
        for word in words:
            if sentence == "":
                sentence = str(word)
            else:
                sentence = sentence + "，" + str(word)
        emb = model.encode(sentence)
        data_tfidf.append(emb)
        data_tfidf_strl.append(str(sentence))
        # data_tfidf.append(sentence)

    # if isinstance(paragraph, list):
    #     words = jieba.analyse.extract_tags(paragraph[0], topK=1)
    #     # words = jieba.analyse.textrank(paragraph[0], topK=2)
        
    #     # cmrc
    #     # 20: {"0": 2098, "-1": 1153}, 
    #     # 15: {"0": 2153, "-1": 1078, "1": 20}, 
    #     # 10: {"0": 2307, "-1": 860, "1": 64, "2": 20},
    #     #  8: {"0": 2378, "-1": 798, "1": 54, "2": 21}, 
    #     #  7: {"0": 2405, "-1": 781, "1": 45, "2": 20}, 
    #     #  6: {"0": 2428, "-1": 761, "1": 34, "2": 18, "3": 10},
    #     # @5: {"0": 2518, "-1": 660, "3": 20, "1": 30, "2": 11, "4": 12}, 
    #     #  4: {"0": 2565, "-1": 642, "1": 27, "2": 17},
    #     #  3: {"0": 2514, "-1": 636, "1": 19, "2": 16, "3": 11, "4": 28, "5": 17, "6": 10}, 
    #     #  2: {"0": 2453, "-1": 591, "4": 23, "1": 61, "2": 20, "3": 11, "8": 20, "5": 20, "10": 11, "11": 10, "7": 11, "6": 10, "9": 10},
    #     # @1: {"-1": 678, "0": 15, "1": 17, "2": 2117, "3": 15, "4": 18, "5": 15, "6": 29, "7": 28, "8": 29, "9": 13, "10": 16, "11": 14, "12": 19, "13": 11, "14": 11, "15": 15, "16": 18, "17": 43, "18": 15, "19": 29, "20": 17, "21": 13, "22": 10, "23": 15, "24": 11, "25": 10, "26": 10} 
    #     #  1: {"-1": 681, "0": 15, "1": 17, "2": 2108, "3": 19, "4": 18, "5": 15, "6": 29, "7": 29, "8": 30, "9": 13, "10": 16, "11": 14, "12": 19, "13": 11, "14": 12, "15": 15, "16": 18, "17": 43, "18": 14, "19": 30, "20": 17, "21": 13, "22": 10, "23": 14, "24": 11, "25": 10, "26": 10}

    #     # drcd
    #     #  5: {"0": 8626, "-1": 343, "1": 35, "2": 10}
    #     #  4: {"0": 8563, "1": 111, "-1": 319, "2": 21}
    #     #  3: {"0": 8569, "-1": 346, "1": 71, "2": 19, "3": 9}
    #     # @2: {"0": 8280, "-1": 403, "1": 116, "3": 35, "2": 20, "5": 27, "4": 24, "8": 16, "6": 37, "7": 12, "9": 44}
    #     # @1: {"0": 51, "1": 6136, "2": 99, "3": 48, "4": 63, "5": 33, "6": 35, "7": 44, "-1": 808, "8": 43, "9": 42, "10": 17, "11": 52, "12": 88, "13": 59, "14": 47, "15": 38, "16": 53, "17": 136, "18": 20, "19": 24, "20": 63, "21": 24, "22": 72, "23": 48, "24": 45, "25": 18, "50": 10, "26": 21, "27": 31, "37": 43, "28": 12, "29": 56, "30": 16, "31": 13, "32": 21, "33": 11, "34": 37, "60": 10, "35": 39, "36": 41, "45": 42, "38": 11, "59": 10, "39": 21, "40": 39, "41": 11, "42": 10, "43": 46, "44": 27, "46": 10, "47": 23, "48": 20, "49": 14, "55": 19, "51": 13, "52": 20, "53": 10, "54": 27, "56": 23, "57": 14, "58": 13, "61": 22, "62": 2}
    #     for word in words:
    #         if sentence == "":
    #             sentence = str(word)
    #         else:
    #             sentence = sentence + "，" + str(word)
    #     emb = model.encode(sentence)
    #     data_tfidf.append(emb)
    # elif isinstance(paragraph, str):
    #     words = jieba.analyse.extract_tags(paragraph, topK=1)
    #     for word in words:
    #         if sentence == "":
    #             sentence = str(word)
    #         else:
    #             sentence = sentence + "，" + str(word)
    #     emb = model.encode(sentence)
    #     data_tfidf.append(emb)
    # else:
    #     print(type(paragraph))

print(type(data_tfidf_strl))
print(type(data_tfidf_strl[0]))
with open(data_check_path, "w", encoding='utf8') as check_file: 
    json.dump(data_tfidf_strl, check_file, ensure_ascii=False)


# print("tfidf len", len(data_tfidf))   # 9014
arr_emb = numpy.array(data_tfidf)
# print("arr shape: ", arr_emb.shape())

# cluster
# fit <class 'numpy.ndarray'>
cluster = DBSCAN(eps=3,min_samples=10).fit(arr_emb)

# scatter plot
# 3d 
# L_sk = PCA(3).fit_transform(arr_emb)
# # print('L_sk.shape:', L_sk.shape)          (1960, 3)   [[a_0 b_0 c_0][a_1 b_1 c_1]...[a_1959 b_1959 c_1959]]
# fig = plt.figure(figsize = (10, 7))
# ax = plt.axes(projection ="3d")
# ax.scatter3D(L_sk[:,0], L_sk[:,1], L_sk[:,2],c=cluster.labels_)
# plt.savefig(fig_path)

# 2d
L_sk = PCA(2).fit_transform(arr_emb)
plt.scatter(L_sk[:,0],L_sk[:,1],c=cluster.labels_, cmap='Spectral')  
plt.savefig(fig_path)

label_list = list(cluster.labels_)
new_list = []    
for i in label_list:
    i = int(i)
    new_list.append(i)

cnt_list = collections.OrderedDict()
for i in new_list:
    if i not in cnt_list:
        cnt_list[i] = 1
    else:
        cnt_list[i] = cnt_list[i] + 1

for idx, val in enumerate(new_list):
    if idx < train_len:
        for article in new_data['data']:
            for paragraph in article['paragraphs']:
                paragraph['title_label'] = val
    else:
        for article in new_data_t['data']:
            for paragraph in article['paragraphs']:
                paragraph['title_label'] = val          

# with open(emb_path, "w", encoding='utf8') as emb_file, \
#     open(cnt_path, "w", encoding='utf8') as cnt_file, \
#     open(data_new_path, "w", encoding='utf8') as n_data_file, \
#     open(data_new_t_path, "w", encoding='utf8') as n_data_t_file:
#     json.dump(new_list, emb_file, ensure_ascii=False)
#     json.dump(cnt_list, cnt_file, ensure_ascii=False)
#     json.dump(new_data, n_data_file, ensure_ascii=False)
#     json.dump(new_data_t, n_data_t_file, ensure_ascii=False)
