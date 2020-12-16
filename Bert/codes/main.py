import os,sys
import numpy as np
import argparse
import keras
import datetime
import glob

#os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE' # set as True on certain remote servers

from keras_bert import  gen_batch_inputs
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.metrics import  f1_score, precision_score, recall_score
from keras.callbacks import TensorBoard

from get_bert_model import *
from data_helper import *
from utils.data_utils import *

parser = argparse.ArgumentParser()
parser.add_argument('--model_type', default="albert", type=str, required=False, help='types of pre-trained models，"bert" means BERT or RoBERTa，"albert"means ALBERT model')
parser.add_argument('--times', default=1, type=int, required=False, help='declear paths to load data or store models ( /data、/models)for each run')
parser.add_argument('--nclass', default=2, type=int, required=False, help='the number of classes for classification tasks')
parser.add_argument('--epoch', default=2, type=int, required=False, help='epoch')
parser.add_argument('--lr', default=1e-5, type=float, required=False, help='original static learning rate')
parser.add_argument('--batch_size', default=2, type=int, required=False, help='batch_size')
parser.add_argument('--dev_size', default=0.1, type=float, required=False, help='size of the validation set split from train set before training')
parser.add_argument('--maxlen', default=512, type=int, required=False, help='the maximum length of input sentences')
parser.add_argument('--pretrained_path', default="D:/codes/Bert_projects/pre_trained_models/albert_tiny_google_zh_489k/", type=str, required=False, help='path of pre-trained model files')
parser.add_argument('--log_name', default="training_info", type=str, required=False, help='name of the log file')
parser.add_argument('--submision_sample_path', default="../subs/submit_example.csv", type=str, required=False, help='预测结果提交文件')
parser.add_argument('--do_pretrain', default=False,action='store_true', required=False, help='choose to pre-train from scratch')
parser.add_argument('--do_train', default=True,action='store_true', required=False, help='choose to train from pretrained BERT ')
parser.add_argument('--do_keep_train', default=False,action='store_true', required=False, help='choose to load from trained checkpoint and contnue training')
parser.add_argument('--do_predict', default=False, action='store_true', required=False, help='choose to test')

args = parser.parse_args()

args.pretrain_data_path = "../data/"+str(args.times)+"/pretrain.tsv" #pre-train corpus is stored in file pretrain.tsv
args.train_path = "../data/"+str(args.times)+"/train.tsv"
args.test_path = "../data/"+str(args.times)+"/test.tsv"
args.sub_path = "../subs/"+str(args.times)+"/"
args.log_path = "../logs/"+str(args.times)+"/"
args.finetuned_path = "../models/"+str(args.times)+"/"
sys.stdout = MyLogger(args.log_path + args.log_name, sys.stdout)

for each in glob.glob(args.pretrained_path+"/*"):
    if ".json" in each:
        args.config_path = each
        continue
    if ".ckpt." in each:
        args.checkpoint_path = each[0:each.find("ckpt")+4]
        continue
    if ".txt" in each:
        args.vocab_path = each

if os.path.exists(args.log_path)==False:
    os.mkdir(args.log_path, mode=0o777)
if os.path.exists(args.sub_path)==False:
    os.mkdir(args.sub_path, mode=0o777)
if os.path.exists(args.finetuned_path)==False:
    os.mkdir(args.finetuned_path, mode=0o777)
    
label2idx = {"O": 0,
             "B-PER": 1, "I-PER": 2,
             "B-LOC": 3, "I-LOC": 4,
             "B-ORG": 5, "I-ORG": 6,
             "B-TIM": 7, "I-TIM": 8,
             "B-COM": 9, "I-COM": 10,
             "B-PRO": 11, "I-PRO": 12,
             }
args.nclass = len(label2idx)  #for NER tasks, nclass means the number of labels
     
class Metrics(Callback):#define callback class，validate and predict after each epoch
    def __init__(self, valid_data,test_data):
        super(Metrics, self).__init__()
        self.validation_data = valid_data
        self.test_data = test_data

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
        
        test_predict = np.argmax(self.model.predict([self.test_data[0],self.test_data[1]],verbose=1), -1)
        test_targ = self.test_data[2]
        if len(test_targ.shape) == 2 and test_targ.shape[1] != 1:
            test_targ = np.argmax(test_targ, -1)
        print("report on test set")
        _test_f1 = f1_score(test_targ, test_predict, average='macro')
        _test_recall = recall_score(test_targ, test_predict, average='macro')
        _test_precision = precision_score(test_targ, test_predict, average='macro')

        logs['test_f1'] = _test_f1
        logs['test_recall'] = _test_recall
        logs['test_precision'] = _test_precision
        
        print(" — test_f1: %f — test_precision: %f — test_recall: %f" % (_test_f1, _test_precision, _test_recall))
        return

#pertrain() method is only available for BERT and RoBERTa models
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

def mrc_train():
    if args.model_type =="bert" or args.model_type =="roberta":
        model = build_mrc_bert(args)
    elif args.model_type =="albert":
        model = build_mrc_albert(args)
     
    sentence_pairs,ans = get_mrc_data(args.train_path) #data是由所有句子组成的一级列表["你是人","你不是人"]
    x_train,x_test, y_train, y_test =train_test_split(sentence_pairs,ans,test_size=args.dev_size, random_state=2020)

    x1_train,x2_train,train_start_points,train_end_points=get_bert_input_mrc(x_train,y_train,args.vocab_path,args.maxlen)#x1、x2
    x1_test,x2_test,test_start_points,test_end_points=get_bert_input_mrc(x_test,y_test,args.vocab_path,args.maxlen)#x1、x2

    checkpoint = ModelCheckpoint('../models/'+str(args.times)+'/model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    #f1 = Metrics(valid_data=[[np.array(x1_test),np.array(x2_test)], [np.array(test_start_points),np.array(test_end_points)]])

    log_dir = os.path.join(
    "mrc_logs",
    "fit",
    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    tb = TensorBoard(log_dir=log_dir,  # log 目录
                     histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                     #                  batch_size=32,     # 用多大量的数据计算直方图
                     write_graph=True,  # 是否存储网络结构图
                     write_grads=True,  # 是否可视化梯度直方图
                     write_images=True,  # 是否可视化参数
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None,
                     update_freq='batch'
                     )
    print(np.array(x1_train).shape)
    print(np.array(x2_train).shape)
    print(np.array(train_start_points).shape)
    print(np.array(train_end_points).shape)
    model.fit([np.array(x1_train),np.array(x2_train)], [np.array(train_start_points),np.array(train_end_points)],
             batch_size=args.batch_size,
             epochs=args.epoch,
             verbose=1,
             sample_weight = None,
             callbacks=[checkpoint,tb],
             validation_data=[[np.array(x1_test),np.array(x2_test)], [np.array(test_start_points),np.array(test_end_points)]]
             )
             
def ner_train():
    if args.model_type =="bert" or args.model_type =="roberta":
        model = build_ner_bert(args)
    elif args.model_type =="albert":
        model = build_ner_albert(args)
     
    data,label = get_ner_data(args,label2idx) #data是由所有句子组成的一级列表["你是人","你不是人"]

    label = keras.utils.to_categorical(label, args.nclass)

    x_train,x_test, y_train, y_test =train_test_split(data,label,test_size=args.dev_size, random_state=0)

    x1_train,x2_train=get_bert_input_ner(x_train,args)#x1、x2
    x1_test,x2_test=get_bert_input_ner(x_test,args)#x1、x2

    checkpoint = ModelCheckpoint('../models/'+str(args.times)+'/model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    f1 = Metrics([np.array(x1_test),np.array(x2_test)], np.array(y_test))
    log_dir = os.path.join(
        "ner_logs",
        "fit",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )

    tb = TensorBoard(log_dir=log_dir,  # log 目录
                     histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                     #                  batch_size=32,     # 用多大量的数据计算直方图
                     write_graph=True,  # 是否存储网络结构图
                     write_grads=True,  # 是否可视化梯度直方图
                     write_images=True,  # 是否可视化参数
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None,
                     update_freq='batch'
                     )
    model.fit([np.array(x1_train),np.array(x2_train)], np.array(y_train),
             batch_size=args.batch_size,
             epochs=args.epoch,
             verbose=1,
             callbacks=[checkpoint,f1,tb],
             validation_data=([np.array(x1_test),np.array(x2_test)], np.array(y_test))
             )  

def cls_train():
    if args.model_type =="bert" or args.model_type =="roberta":
        model = build_cls_bert(args)
    elif args.model_type =="albert":
        model = build_cls_albert(args)
        
    data,label = get_cls_data(args.train_path) #data: ["hi you","all right"]
    
    if args.nclass > 2:
        label=pd.get_dummies(pd.DataFrame(label)) 
    print(label[0:10],data[0:10])
    
    x_train,x_val, y_train, y_val =train_test_split(data,label,test_size=args.dev_size, random_state=2020)

    x1_train,x2_train=get_bert_input_cls(x_train,args.vocab_path,args.maxlen)#x1、x2
    x1_val,x2_val=get_bert_input_cls(x_val,args.vocab_path,args.maxlen)#x1、x2

    test_data,test_label = get_cls_data(args.test_path)
    x1_test,x2_test=get_bert_input_cls(test_data,args.vocab_path,args.maxlen)

    checkpoint = ModelCheckpoint('../models/'+str(args.times)+'/model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='auto')
    #f1 = Metrics(valid_data=[np.array(x1_val),np.array(x2_val), np.array(y_val)],test_data = [np.array(x1_test),np.array(x2_test), np.array(test_label)] )
    log_dir = os.path.join(
        "cls_logs",
        "fit",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
    tb = TensorBoard(log_dir=log_dir,  # log path
                     histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                     batch_size=32,     # 用多大量的数据计算直方图
                     write_graph=True,  # 是否存储网络结构图
                     write_grads=True,  # 是否可视化梯度直方图
                     write_images=True,  # 是否可视化参数
                     embeddings_freq=0,
                     embeddings_layer_names=None,
                     embeddings_metadata=None,
                     update_freq='batch'
                     )
    model.fit([np.array(x1_train),np.array(x2_train)], np.array(y_train),
             batch_size=args.batch_size,
             epochs=args.epoch,
             verbose=1,
             callbacks=[checkpoint,tb],
             validation_data=[[np.array(x1_val),np.array(x2_val)], np.array(y_val)]
             )
    #model.save("../models/"+str(args.times)+"/trained.h5")

def keep_train():
    if args.model_type =="bert" or args.model_type =="roberta":
        model = build_bert(args)
        model.load_weights(args.finetuned_path+"trained.h5" , by_name = True, skip_mismatch = True)#only load layers with correct name
    elif args.model_type =="albert":
        model = build_albert(args)
        model.load_weights(args.finetuned_path+"trained.h5" , by_name = True, skip_mismatch = True)
    #model.load_weights(args.finetuned_path+"bert.h5")

    data,label = get_finetune_data(args.train_path)

    print(label[0:10],data[0:10])
    x_train,x_val, y_train, y_val =train_test_split(data,label,test_size=args.dev_size, random_state=2020)

    x1_train,x2_train=get_bert_input(x_train,args.vocab_path,args.maxlen)#x1、x2
    x1_val,x2_val=get_bert_input(x_val,args.vocab_path,args.maxlen)#x1、x2

    test_data,test_label = get_finetune_data(args.test_path)
    x1_test,x2_test=get_bert_input(test_data,args.vocab_path,args.maxlen)
    
    checkpoint = ModelCheckpoint('../models/'+str(args.times)+'/keep_train_weights.{epoch:03d}-{val_f1:.4f}.h5', monitor='val_f1', verbose=1, save_best_only=True, mode='auto')
    early_stopping = EarlyStopping(monitor='val_f1', min_delta=0, patience=1, verbose=1)
    metrics = Metrics(valid_data=[np.array(x1_val),np.array(x2_val), np.array(y_val)],test_data = [np.array(x1_test),np.array(x2_test), np.array(test_label)] )
    
    model.fit([np.array(x1_train),np.array(x2_train)], np.array(y_train),
             batch_size=args.batch_size,
             epochs=args.epoch,
             verbose=1,
             callbacks=[early_stopping,metrics,checkpoint],
             validation_data=[[np.array(x1_val),np.array(x2_val)], np.array(y_val)]
             )
    model.save("../models/"+str(args.times)+"/keep_trained.h5")

def predict(): #predict function changes according to different training tasks
    if args.model_type =="bert" or args.model_type =="roberta":
        model = build_bert(args)
    elif args.model_type =="albert":
        model = build_albert(args)
    model.load_weights(args.finetuned_path + "trained.h5")

    data,label = get_finetune_data(args.test_path)
    x1_train,x2_train=get_bert_input(data,args.vocab_path,args.maxlen)

    predictions = model.predict([np.array(x1_train),np.array(x2_train)],verbose=1)
    detailed_predictions(args,predictions)#record numerical probability data of each prediction for potential merge operation
    final_result = predictions.argmax(axis=-1)
    write_csv(args,final_result)



if __name__ == "__main__":
    print(args)
    if args.do_pretrain == True:
       pretrain()
    if args.do_train == True:
        # different _train() function can be applied according to tasks.
       mrc_train()
    if args.do_keep_train == True:
       keep_train()
    if args.do_predict == True:
       predict()
