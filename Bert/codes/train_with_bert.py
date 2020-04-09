import os
import numpy as np
import argparse
import keras
import glob

#os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'#使用服务器添加的指令

from keras_bert import  gen_batch_inputs
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import  f1_score, precision_score, recall_score

from get_bert_model import *
from data_helper import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default="albert", type=str, required=False, help='预训练模型类型，bert或roberta均使用默认"bert"即可，若为albert需要特殊声明')
parser.add_argument('--times', default=1, type=int, required=False, help='选择/data、/models下不同序号文件夹进行本次训练')
parser.add_argument('--nclass', default=3, type=int, required=False, help='几分类')
parser.add_argument('--epoch', default=2, type=int, required=False, help='训练批次')
parser.add_argument('--lr', default=1e-5, type=float, required=False, help='训练批次')
parser.add_argument('--batch_size', default=2, type=int, required=False, help='batch_size')
parser.add_argument('--dev_size', default=0.1, type=float, required=False, help='test_size')
parser.add_argument('--maxlen', default=512, type=int, required=False, help='训练样本最大句子长度')
parser.add_argument('--pretrained_path', default="D:/codes/Bert_projects/pre_trained_models/albert_tiny_google_zh_489k/", type=str, required=False, help='预训练模型保存目录')
parser.add_argument('--submision_sample_path', default="../subs/submit_example.csv", type=str, required=False, help='预测结果提交文件')
parser.add_argument('--do_pretrain', default=False,action='store_true', required=False, help='是否预训练')
parser.add_argument('--do_train', default=False,action='store_true', required=False, help='是否训练')
parser.add_argument('--do_keep_train', default=True,action='store_true', required=False, help='是否训练')
parser.add_argument('--do_predict', default=True, action='store_true', required=False, help='提交测试')

args = parser.parse_args()

args.pretrain_data_path = "../data/"+str(args.times)+"/pretrain.tsv"
args.train_path = "../data/"+str(args.times)+"/train.tsv"
args.test_path = "../data/"+str(args.times)+"/test.tsv"
args.sub_path = "../subs/"+str(args.times)+"/"
args.finetuned_path = "../models/"+str(args.times)+"/"

for each in glob.glob(args.pretrained_path+"/*"):
    if ".json" in each:
        args.config_path = each
        continue
    if ".ckpt." in each:
        args.checkpoint_path = each[0:each.find("ckpt")+4]
        continue
    if ".txt" in each:
        args.vocab_path = each

if os.path.exists(args.sub_path)==False:
    os.mkdir(args.sub_path, mode=0o777)
if os.path.exists(args.finetuned_path)==False:
    os.mkdir(args.finetuned_path, mode=0o777)

class Metrics(Callback):
    def __init__(self, valid_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        val_predict = np.argmax(self.model.predict([self.validation_data[0],self.validation_data[1]]), -1)
        val_targ = self.validation_data[2]
        if len(val_targ.shape) == 2 and val_targ.shape[1] != 1:
            val_targ = np.argmax(val_targ, -1)

        _val_f1 = f1_score(val_targ, val_predict, average='macro')
        _val_recall = recall_score(val_targ, val_predict, average='macro')
        _val_precision = precision_score(val_targ, val_predict, average='macro')

        logs['val_f1'] = _val_f1
        logs['val_recall'] = _val_recall
        logs['val_precision'] = _val_precision
        print(" — val_f1: %f — val_precision: %f — val_recall: %f" % (_val_f1, _val_precision, _val_recall))
        return
#只有bert/roberta支持继续的pertrain
def pretrain():
    sentence_pairs = get_pretrain_data(args)
    token_dict = {}
    with codecs.open(args.vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    token_list = list(token_dict.keys())
    model = pretrain_bert(args,training= True)
    
    def _generator():
        while True:
            for pair in sentence_pairs:
                yield gen_batch_inputs(
                    [pair],
                    token_dict,
                    token_list,
                    seq_len=512,
                    mask_rate=0.3,
                    swap_sentence_rate=0,
                )

    model.fit_generator(
        generator=_generator(),
        steps_per_epoch=900000,
        epochs=5,
        validation_data=_generator(),
        verbose=1,
        validation_steps=10000,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor='val_loss', patience=1),
            ModelCheckpoint('../models/'+str(args.times)+'/weights.{epoch:03d}-{val_loss:.4f}.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
        ],
    )    

    model.save("../models/"+str(args.times)+"/pretrained_bert.h5")

def train():
    if args.model_type =="bert" or args.model_type =="roberta":
        model = build_bert(args)
    elif args.model_type =="albert":
        model = build_albert(args)
    data,label = get_finetune_data(args.train_path) #data是由所有句子组成的一级列表["你是人","你不是人"]

    print(label[0:10],data[0:10])
    x_train,x_test, y_train, y_test =train_test_split(data,label,test_size=args.dev_size, random_state=2020)

    x1_train,x2_train=get_bert_input(x_train,args.vocab_path,args.maxlen)#x1、x2
    x1_test,x2_test=get_bert_input(x_test,args.vocab_path,args.maxlen)#x1、x2



    checkpoint = ModelCheckpoint('../models/'+str(args.times)+'/weights.{epoch:03d}-{val_f1:.4f}.h5', monitor='val_f1', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_f1', min_delta=0, patience=1, verbose=1)
    metrics = Metrics(valid_data=[[np.array(x1_test),np.array(x2_test)], np.array(y_test)])
    model.fit([np.array(x1_train),np.array(x2_train)], np.array(y_train),
             batch_size=args.batch_size,
             epochs=args.epoch,
             verbose=1,
             callbacks=[checkpoint,early_stopping,metrics],
             validation_data=[[np.array(x1_test),np.array(x2_test)], np.array(y_test)]
             )
    model.save("../models/"+str(args.times)+"/trained.h5")

def keep_train():
    if args.model_type =="bert" or args.model_type =="roberta":
        model = build_bert(args)
        model.load_weights(args.finetuned_path+"trained.h5" , by_name = True, skip_mismatch = True)#只加载对应名存在的层
    elif args.model_type =="albert":
        model = build_albert(args)
        model.load_weights(args.finetuned_path+"trained.h5" , by_name = True, skip_mismatch = True)#只加载对应名存在的层
    #model.load_weights(args.finetuned_path+"bert.h5")

    data,label = get_finetune_data(args.train_path) #data是由所有句子组成的一级列表["你是人","你不是人"]

    print(label[0:10],data[0:10])
    x_train,x_test, y_train, y_test =train_test_split(data,label,test_size=args.dev_size, random_state=2020)

    x1_train,x2_train=get_bert_input(x_train,args.vocab_path,args.maxlen)#x1、x2
    x1_test,x2_test=get_bert_input(x_test,args.vocab_path,args.maxlen)#x1、x2

    checkpoint = ModelCheckpoint('../models/'+str(args.times)+'/keep_train_weights.{epoch:03d}-{val_f1:.4f}.h5', monitor='val_f1', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_f1', min_delta=0, patience=1, verbose=1)
    metrics = Metrics(valid_data=[[np.array(x1_test),np.array(x2_test)], np.array(y_test)])    
    
    model.fit([np.array(x1_train),np.array(x2_train)], np.array(y_train),
             batch_size=args.batch_size,
             epochs=args.epoch,
             verbose=1,
             callbacks=[early_stopping,metrics,checkpoint],
             validation_data=[[np.array(x1_test),np.array(x2_test)], np.array(y_test)]
             )
    model.save("../models/"+str(args.times)+"/keep_trained.h5")
def predict():
    if args.model_type =="bert" or args.model_type =="roberta":
        model = build_bert(args)
    elif args.model_type =="albert":
        model = build_albert(args)
    model.load_weights(args.finetuned_path + "trained.h5")

    data,label = get_finetune_data(args.test_path)
    x1_train,x2_train=get_bert_input(data,args.vocab_path,args.maxlen)

    predictions = model.predict([np.array(x1_train),np.array(x2_train)],verbose=1)
    detailed_predictions(predictions)#记录softmax后预测概率原始值
    final_result = predictions.argmax(axis=-1)
    write_csv(final_result)

def write_csv(result):#将预测结果写入/sub目录下csv文件
    result = [-1 if each==2 else each for each in result]
    id = pd.read_csv(args.submision_sample_path)[["id"]]
    result = pd.DataFrame(result,columns = ["y"])
    result = id.join(result)
    result.to_csv(args.sub_path+"result.csv",index = False)

def detailed_predictions(predictions):
    f = open("../subs/"+str(args.times)+"/predictions.txt","w",encoding = "utf8")
    for each in predictions:
        tem = [str(i) for i in each]
        f.write(" ".join(tem))
        f.write("\n")
    f.close()

if __name__ == "__main__":
    if args.do_pretrain == True:
       pretrain()
    if args.do_train == True:
       train()
    if args.do_keep_train == True:
       keep_train()
    if args.do_predict == True:
       predict()
