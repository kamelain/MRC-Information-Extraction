# coding=utf-8
from __future__ import absolute_import, division, print_function

import collections
import logging
import math

import numpy as np
import torch
from pytorch_transformers import (WEIGHTS_NAME, BertConfig,
                                  BertForQuestionAnswering, BertTokenizer)
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from utils import (get_answer, input_to_squad_example,
                   squad_examples_to_features, to_list)

RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

max_seq_length = 384
doc_stride = 128
do_lower_case = True
max_query_length = 64
n_best_size = 20
max_answer_length = 30
device = 'cpu'
model_path = '/share/nas167/chinyingwu/nlp/dialog/corpus/chinese_wwm_L-12_H-768_A-12'

def predict(passage :str,question :str):
    example = input_to_squad_example(passage,question)
    features = squad_examples_to_features(example, tokenizer, max_seq_length, doc_stride, max_query_length)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids,
                            all_example_index)
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=1)
    all_results = []
    for batch in eval_dataloader:
        batch = tuple(t.to( device) for t in batch)
        with torch.no_grad():
            inputs = {'input_ids':      batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2]  
                    }
            example_indices = batch[3]
            outputs =  model(**inputs)

        for i, example_index in enumerate(example_indices):
            eval_feature = features[example_index.item()]
            unique_id = int(eval_feature.unique_id)
            result = RawResult(unique_id    = unique_id,
                                start_logits = to_list(outputs[0][i]),
                                end_logits   = to_list(outputs[1][i]))
            all_results.append(result)
    answer = get_answer(example,features,all_results, n_best_size, max_answer_length, do_lower_case)
    return answer

if __name__ == "__main__":
    config = BertConfig.from_pretrained(model_path + "/bert_config.json")
    tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case)
    model = BertForQuestionAnswering.from_pretrained(model_path, from_tf=False, config=config)
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    model.to(device)
    model.eval()

    doc = "硫是一種化學元素，原子序數是16。硫是一種非常常見的無味無臭的非金屬，純的硫是黃色的晶體，又稱做硫磺。硫有許多不同的化合價，常見的有-2、0、+4、+6等。在自然界中它經常以硫化物或硫酸鹽的形式出現，尤其在火山地區純的硫也在自然界出現。對所有的生物來說，硫都是一種重要的必不可少的元素，它是多種胺基酸的組成部分，尤其是大多數蛋白質的組成部分。它主要被用在肥料中，也廣泛地被用在火藥、潤滑劑、殺蟲劑和抗真菌劑中。純的硫呈淺黃色，質地柔軟，輕。與氫結成有毒化合物硫化氫後有一股臭味。硫燃燒時的火焰是藍色的，並散發出一種特別的硫磺味。硫不溶於水但溶於二硫化碳。硫最常見的化學價是-2、+2、+4和+6。在所有的物態中，硫都有不同的同素異形體，這些同素異形體的相互關係還沒有被完全理解。晶體的硫可以組成一個由八個原子組成的環。硫有兩種晶體形式：斜方晶八面體和單斜棱晶體，前者在室溫下比較穩定。"
    q = '為何硫又稱硫磺？'
    ans = predict(doc, q)
    print(ans)
