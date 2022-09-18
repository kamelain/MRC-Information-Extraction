# MRC-Information-Extraction
A Study on Contextualized Language Modeling for Machine Reading Comprehension 

基於 [BERT](https://github.com/google-research/bert)、[BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm) 以及 [MacBERT](https://github.com/ymcui/MacBERT) 的官方資源修改。

適用於中文機器閱讀理解：繁體中文 [DRCD](https://github.com/DRCSolutionService/DRCD)、簡體中文 [CMRC](https://github.com/ymcui/cmrc2018)。

本程式碼為論文 [A Study on Contextualized Language Modeling for Machine Reading Comprehension](https://aclanthology.org/2021.rocling-1.7/) 以及 [A Study on the Information Extraction and Knowledge Injection for Machine Reading Comprehension](https://etds.lib.ntnu.edu.tw/thesis/detail/c7f11bb51318d02b9874ae5429b6eb82/?seq=1) 於單輪機器閱讀理解的實作部分。包括 Fine-tune 於 BERT/BERT-wwm/MacBERT 結果、加入 Information Extration 資訊結果，以及 N-best 答案進行 Reranking 的 Ensemble 方法與結果。

>Clutering Strategies
![Clustering Strategies](https://github.com/kamelain/MRC-Information-Extraction/blob/main/Screen%20Shot%202022-09-18%20at%208.31.49%20PM.png)


### Conda 環境設定檔

>* tf.yml 模型訓練環境
>* prep.yml 資料預處理環境（Clustering、Ensemble）


### 使用

* Information Extraction

```
bash cls/run_cls.sh
```

* Train & Prediction

```
bash run.sh
```

* Evaluate

```
bash eval.sh
```

### Fine-tuning DRCD 

>* output
>* output_wwm
>* output_mac

### Fine-tuning CMRC

>* output
>* output_c_wwm
>* output_c_mac

### Information Extraction 

>* cls


>Result
![result](https://github.com/kamelain/MRC-Information-Extraction/blob/main/Screen%20Shot%202022-09-18%20at%208.31.08%20PM.png)
