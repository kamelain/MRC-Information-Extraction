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

model_path = 'model/'

class QA:
    def __init__(self,model_path: str):
        # model_path = '/share/nas167/chinyingwu/nlp/dialog/corpus/chinese_L-12_H-768_A-12'
        # output_path = '/output_chinese'
        self.max_seq_length = 384
        self.doc_stride = 128
        self.do_lower_case = True
        self.max_query_length = 64
        self.n_best_size = 20
        self.max_answer_length = 30
        self.model, self.tokenizer = self.load_model(model_path)
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.model.to(self.device)
        self.model.eval()


    def load_model(self, model_path: str,do_lower_case=False):
        config = BertConfig.from_pretrained(model_path + "/bert_config.json")
        tokenizer = BertTokenizer.from_pretrained(model_path, do_lower_case=do_lower_case)
        model = BertForQuestionAnswering.from_pretrained(model_path, from_tf=False, config=config)
    
        # from transformers import BertConfig, BertForSequenceClassification, BertTokenizer
        # model_config, model_class, model_tokenizer = (BertConfig, BertForSequenceClassification, BertTokenizer)
        # config = model_config.from_pretrained(config_file_path,num_labels = num_labels)
        # model = BertForQuestionAnswering.from_pretrained(output_path, from_tf=bool('.ckpt' in 'model'), config=config)
        # tokenizer = model_tokenizer(vocab_file=vocab_file_path)

        return model, tokenizer
    
    def predict(self,passage :str,question :str):
        example = input_to_squad_example(passage,question)
        features = squad_examples_to_features(example,self.tokenizer,self.max_seq_length,self.doc_stride,self.max_query_length)
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
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'token_type_ids': batch[2]  
                        }
                example_indices = batch[3]
                outputs = self.model(**inputs)

            for i, example_index in enumerate(example_indices):
                eval_feature = features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                result = RawResult(unique_id    = unique_id,
                                    start_logits = to_list(outputs[0][i]),
                                    end_logits   = to_list(outputs[1][i]))
                all_results.append(result)
        answer = get_answer(example,features,all_results,self.n_best_size,self.max_answer_length,self.do_lower_case)
        return answer


# def predict(doc, q):
#     # from bert import QA
#     model = QA('/share/nas167/chinyingwu/nlp/dialog/corpus/chinese_L-12_H-768_A-12')

#     # doc = "Victoria has a written constitution enacted in 1975, but based on the 1855 colonial constitution, passed by the United Kingdom Parliament as the Victoria Constitution Act 1855, which establishes the Parliament as the state's law-making body for matters coming under state responsibility. The Victorian Constitution can be amended by the Parliament of Victoria, except for certain 'entrenched' provisions that require either an absolute majority in both houses, a three-fifths majority in both houses, or the approval of the Victorian people in a referendum, depending on the provision."
#     # q = 'When did Victoria enact its constitution?'
#     # doc = "硫是一種化學元素，原子序數是16。硫是一種非常常見的無味無臭的非金屬，純的硫是黃色的晶體，又稱做硫磺。硫有許多不同的化合價，常見的有-2、0、+4、+6等。在自然界中它經常以硫化物或硫酸鹽的形式出現，尤其在火山地區純的硫也在自然界出現。對所有的生物來說，硫都是一種重要的必不可少的元素，它是多種胺基酸的組成部分，尤其是大多數蛋白質的組成部分。它主要被用在肥料中，也廣泛地被用在火藥、潤滑劑、殺蟲劑和抗真菌劑中。純的硫呈淺黃色，質地柔軟，輕。與氫結成有毒化合物硫化氫後有一股臭味。硫燃燒時的火焰是藍色的，並散發出一種特別的硫磺味。硫不溶於水但溶於二硫化碳。硫最常見的化學價是-2、+2、+4和+6。在所有的物態中，硫都有不同的同素異形體，這些同素異形體的相互關係還沒有被完全理解。晶體的硫可以組成一個由八個原子組成的環。硫有兩種晶體形式：斜方晶八面體和單斜棱晶體，前者在室溫下比較穩定。"
#     # q = '為何硫又稱硫磺？'
#     # 純的硫是黃色的晶體
#     # q = "硫磺味會因為硫發生什麼事情而產生？"

#     answer = model.predict(doc,q)

#     print(answer['answer'])
#     # 1975
#     print(answer.keys())
#     # dict_keys(['answer', 'start', 'end', 'confidence', 'document']))

# if __name__ == "__main__":    
#     doc = "硫是一種化學元素，原子序數是16。硫是一種非常常見的無味無臭的非金屬，純的硫是黃色的晶體，又稱做硫磺。硫有許多不同的化合價，常見的有-2、0、+4、+6等。在自然界中它經常以硫化物或硫酸鹽的形式出現，尤其在火山地區純的硫也在自然界出現。對所有的生物來說，硫都是一種重要的必不可少的元素，它是多種胺基酸的組成部分，尤其是大多數蛋白質的組成部分。它主要被用在肥料中，也廣泛地被用在火藥、潤滑劑、殺蟲劑和抗真菌劑中。純的硫呈淺黃色，質地柔軟，輕。與氫結成有毒化合物硫化氫後有一股臭味。硫燃燒時的火焰是藍色的，並散發出一種特別的硫磺味。硫不溶於水但溶於二硫化碳。硫最常見的化學價是-2、+2、+4和+6。在所有的物態中，硫都有不同的同素異形體，這些同素異形體的相互關係還沒有被完全理解。晶體的硫可以組成一個由八個原子組成的環。硫有兩種晶體形式：斜方晶八面體和單斜棱晶體，前者在室溫下比較穩定。"
#     q = '為何硫又稱硫磺？'
#     # 純的硫是黃色的晶體
#     # q = "硫磺味會因為硫發生什麼事情而產生？"
#     predict(doc,q)