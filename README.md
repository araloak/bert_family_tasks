# bert_family_classification
支持Roberta、albert、bert以及转化为tf版本的ernie等bert大家族所有预训练模型的加载、text_classification finetune与预测。

支持分段设置学习率，提高模型表现。

支持继续在unlabelled_data中继续进行预训练pertrain（albert除外）

支持冻结不同attention层以优化模型效果（albert除外）


************************************************************************************************************************

/stripts目录下保存在服务器进行训练的命令

/data中每个数字子目录保存每次对模型或者数据进行调整之后的训练数据，便于模型调优

/models中每个数字子目录保存每次进行调优后的模型

/sub中每个数字子目录保存每次调优后的predict提交csv文件

************************************************************************************************************************

roberta系列预训练模型:https://github.com/brightmart/roberta_zh

albert系列预训练模型:https://github.com/bojone/albert_zh

tf版ERNIE预训练模型:https://github.com/ArthurRizar/tensorflow_ernie

************************************************************************************************************************
感谢各位大佬的贡献~

客官若是满意的话，请star支持一下吧( ￣▽￣)σ
