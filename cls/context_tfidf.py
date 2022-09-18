# coding:utf-8  
import json, numpy, collections, time
import jieba.analyse
from sklearn.feature_extraction.text import CountVectorizer  

data_path = "../cmrc2018/train.json"
data_t_path = "../cmrc2018/dev.json"
data_new_path = "dataset_ext/cmrc2018/train.json"
data_new_t_path = "dataset_ext/cmrc2018/dev.json"
data_check_path = "dataset_ext/cmrc2018/check.json"
data_check2_path = "dataset_ext/cmrc2018/check2.json"

# data_path = "../DRCD/DRCD_training.json"
# data_t_path = "../DRCD/DRCD_test.json"
# data_new_path = "dataset_cls/DRCD/DRCD_training.json"
# data_new_t_path = "dataset_cls/DRCD/DRCD_test.json"

# set dic for jieba

with open(data_path, encoding = 'utf-8') as data_json, open(data_t_path, encoding = 'utf-8') as data_t_json:
    dataset = collections.OrderedDict(json.load(data_json))
    dataset_t = collections.OrderedDict(json.load(data_t_json))
new_data = dataset
new_data_t = dataset_t
data_paragraph = data_tfidf = data_keyword = []
train_len = 0

# for article in dataset['data']:
#     for paragraph in article['paragraphs']:
#         data_paragraph.append(paragraph['context'])

# train_len = len(data_paragraph)

# for article in dataset_t['data']:
#     for paragraph in article['paragraphs']:
#         data_paragraph.append(paragraph['context'])

# Extract context from data['data']['paragraph']['context']
new_list = [[paragraph["context"] for paragraph in article['paragraphs']] for article in dataset['data']]
test_list = [[paragraph["context"] for paragraph in article['paragraphs']] for article in dataset_t['data']]
new_list = new_list + test_list
# print(type(new_list))
new_list = list(new_list)
# print(data_paragraph)
# print("len: ",len(data_paragraph))
# time_s = time.time()
# str = data_paragraph[2]
# print(str)
# print(jieba.analyse.extract_tags(str, topK=20))
# time_f = time.time()
# print('time:', time_f-time_s)

# with open(data_check_path, "w", encoding='utf8') as data_check_file:
#     json.dump(data_paragraph, data_check_file, ensure_ascii=False)

# print(data_paragraph)
# print(jieba.analyse.extract_tags(data_paragraph[0], topK=20))
for paragraph in new_list:
    sentence = str()
    if isinstance(paragraph, list):
        words = jieba.analyse.extract_tags(paragraph[0], topK=20)
        for word in words:
            if sentence == "":
                sentence = str(word)
            else:
                sentence = sentence + "，" + str(word)
        data_tfidf.append(sentence)
        # data_keyword.append(words)
    else:
        print(type(paragraph))
    # elif isinstance(paragraph, str):
    #     words = jieba.analyse.extract_tags(paragraph, topK=20)
    #     print(type(words))
    #     for word in words:
    #         sentence = sentence +"，"+ str(word)
    #         print(type(word))
    #     data_keyword.append(words)
    # data_tfidf.append(sentence) 
# print(data_tfidf[0])

# with open(data_check_path, "w", encoding='utf8') as data_check_file:
#     json.dump(data_keyword, data_check_file, ensure_ascii=False)
with open(data_check2_path, "w", encoding='utf8') as data_check_file2:
    json.dump(data_tfidf, data_check_file2, ensure_ascii=False)


# separate by jieba 
# for par in data_paragraph:
#     temp = " "
#     for key in jieba.analyse.extract_tags(par, topK=8):
#         temp = temp + " " + key
#     data_tfidf.append(temp)
# print(data_tfidf[0])

    # words = jieba.cut(par)
    # str = ""
    # for word in words:
    #     str = str + " " + word
    #     # print(str)
    # data_sliced.append(str)
# print(data_sliced[0])




# 2403+848
# 8014+1000
for idx, val in enumerate(data_tfidf):
    if idx < train_len:
        for article in new_data['data']:
            for paragraph in article['paragraphs']:
                paragraph['context_ext'] = val
    else:
        for article in new_data_t['data']:
            for paragraph in article['paragraphs']:
                paragraph['context_ext'] = val
print("run")
with open(data_new_path, "w", encoding='utf8') as n_data_file, \
    open(data_new_t_path, "w", encoding='utf8') as n_data_t_file:
    json.dump(new_data, n_data_file, ensure_ascii=False)
    json.dump(new_data_t, n_data_t_file, ensure_ascii=False)

    