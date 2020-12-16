BERT family tasks
===========================
![](https://img.shields.io/badge/keras-tf.keras-blue.svg)
[[中文]](https://github.com/stupidHIGH/bert_family_classification/blob/master/README.zh-CN.md)

Demos of multiple NLP task pipelines.

Implementations are Based on [keras_bert](#https://github.com/CyberZHG/keras-bert).

Support various Transformer-Encoder style pre-trained models. 

In each run, models, data and log files are stored seperately to support massive repeatation of training.

****
# Features

- RoBERTa、ALBERT、BERT and tf version ERNIE are supported
- CLS(classification tasks)、NER(Named Entity Recognition)、MRC(Machine Reading Comprehension) tasks are supported.(prediction needs adjustment according to tasks)
- Dynamically change learning rate
- Continue pre-train on domain data(except for ALBERT)
- Freeze certain encoder layers during training(except for ALBERT)
- Extract output of needed encoder layers and make modifications on downstream architecture
- Provide multiple **loss functions** for unbalanced data sets
- Support tensorboard（tensorboard original codes need to be modified，reference: https://www.jianshu.com/p/9da54361d289）
- Record training information and hyperparameter settings in log files

************************************************************************************************************************

# Usage

- Select needed train functions(cls_train, ner_train,mrc_train) in main.py and modify predict function(model can be automatically tested on test set during training after each epoch in training functions) before performing tasks.

```powershell
python main.py \
  --times 1 \
  --pretrained_path D:/codes/Bert_projects/pre_trained_models/albert_tiny_google_zh_489k/ \
  --log_name training_info \
  --epoch 2 \
  --batch_size 1 \
  --maxlen 512 \
  --do_train  \
  --nclass 3 \
```

************************************************************************************************************************

# Pre-trained Models

RoBERTa: https://github.com/brightmart/roberta_zh

ALBERT: https://github.com/bojone/albert_zh

tf version ERNIE: https://github.com/ArthurRizar/tensorflow_ernie

************************************************************************************************************************

# License
[MIT](./LICENSE)

Forks and Stars are welcomed:blush:.​
