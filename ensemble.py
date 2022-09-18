# -*- coding: utf8 -*-
import os
import json
import collections

wwm_path = "cls/output_ext/output_ext2/nbest_predictions.json"
mac_path = "output_mac/nbest_predictions.json"
n_best_path = "output_ensemble/drcd_mac_macP2/nbest_predictions.json"
predict_path = "output_ensemble/drcd_mac_macP2/predictions.json"

predict = collections.OrderedDict()
n_best = collections.OrderedDict()

# load prediction files
with open(wwm_path, encoding = 'utf-8') as wwm_json:
    wwm = collections.OrderedDict(json.load(wwm_json))
with open(mac_path, encoding = 'utf-8') as mac_json:
    mac = collections.OrderedDict(json.load(mac_json))

# average
for key_wwm, value_wwm in wwm.items():
    for key_mac, value_mac in mac.items():
        if key_wwm == key_mac:
            prob = 0
            for ans_wwm in value_wwm:
                for ans_mac in value_mac:
                    if ans_wwm["text"] == ans_mac["text"]:
                        temp = {}     
                        # text = ans_mac["text"].encode('utf-8').decode('utf-8')
                        # text = ans_mac["text"].decode("unicode-escape")
                        text = ans_mac["text"]
                        temp["text"] = text
                        temp["probability"] = (ans_wwm["probability"]+ans_mac["probability"])/2
                        if key_wwm not in n_best:
                            n_best[key_wwm] = []
                            n_best[key_wwm].append(temp)   
                        else:
                            n_best[key_wwm].append(temp)  

                        if key_wwm not in predict:
                            predict[key_wwm] = {}
                            predict[key_wwm] = temp["text"]
                            prob = temp["probability"]
                        elif temp["probability"] > prob:
                            predict[key_wwm] = temp["text"]
                            prob = temp["probability"]


                if key_wwm not in predict:
                    # print(key_wwm)
                    max_wwm = max(value_wwm, key=lambda x:x['probability'])
                    max_mac = max(value_mac, key=lambda x:x['probability'])
                    
                    if max_wwm['probability'] > max_mac['probability']:
                        predict[key_wwm] = max_wwm['text']
                    else:
                        predict[key_wwm] = max_mac['text']

# output file
with open(predict_path, "w", encoding='utf8') as pred_file:
    json.dump(predict, pred_file, ensure_ascii=False)
with open(n_best_path, "w", encoding='utf8') as nb_file:
    json.dump(n_best, nb_file, ensure_ascii=False)