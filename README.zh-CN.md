BERT family tasks
===========================
[英文](https://github.com/stupidHIGH/bert_family_classification/blob/master/README.md)
多种NLP任务流水线的例子，为高阶更灵活的预训练模型使用提供一定接口，适合BERT新手进行上手练习。
实现基于[keras_bert](#https://github.com/CyberZHG/keras-bert)
支持多种Transformer-Encoder结构的预训练模型加载、训练。
便于多次反复训练，单独保存每次训练数据、模型、日志等文件。


# 支持

- 支持Roberta、albert、bert以及转化为tf版本的ernie等bert大家族所有预训练模型的加载、完成cls、ner、mrc三种经典任务（部分predict代码需要自行调整，train没有大问题）。

- 支持分段设置学习率，提高模型表现。


- 支持继续在domain数据中继续进行pertrain（albert除外）


- 支持冻结不同encoder层以优化模型效果（albert除外）


- 支持提取不同encoder层的输出并在此基础上修改网络结构


- 提供自定义loss和callback函数，便于使用者在此基础上调整


- 支持对于不均衡样本二、多分类的focal loss函数


- 支持tensorboard调用（需要修改tensorboard部分源码，方法参考https://www.jianshu.com/p/9da54361d289），实现训练可视化
- 在log文件中保存训练信息和每次执行的超参数信息

************************************************************************************************************************

# 使用

- 首先在main.py文件中选择不同任务使用的train函数（cls_train, ner_train,mrc_train）并且修改predict函数用于直接加载模型测试（train函数中调用Callback类即可支持训练时每个epoch结束进行val和test）

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

Pre-trained Models

roberta系列预训练模型:https://github.com/brightmart/roberta_zh

albert系列预训练模型:https://github.com/bojone/albert_zh

tf版ERNIE预训练模型:https://github.com/ArthurRizar/tensorflow_ernie

************************************************************************************************************************

# License
[MIT](./LICENSE)

欢迎Forks和Stars:blush:。
