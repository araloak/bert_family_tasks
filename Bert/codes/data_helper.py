import pandas as pd
import codecs
import re

from langconv import *#繁体字转化为简体字
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, extract_embeddings
from keras.preprocessing import sequence

def Traditional2Simplified(sentence):
    '''
    将sentence中的繁体字转为简体字
    :param sentence: 待转换的句子
    :return: 将句子中繁体字转换为简体字之后的句子
    '''
    sentence = Converter('zh-hans').convert(sentence)
    return sentence

def get_cls_data(data_path):
    raw_data=open(data_path,encoding='utf8').read().split('\n')
    label=[i.split("\t")[0] for i in raw_data][0:-1] #最后一个为空行
    processed_test_label = []
    for each in label:
        if each == "-1":
            processed_test_label.append("2")
        else:
            processed_test_label.append(each)    
    corpus=[i[2:].strip() for i in raw_data][0:-1]#最后一个为空行

    return corpus,processed_test_label

def get_ner_data(args,label2idx):
    datas, labels = [], []
    example_path = "D:/codes/Bert_projects/keras_bert_NER/data/examples/example.train"
    #with open(args.train_path, encoding='utf-8') as fr:
    with open(example_path, encoding='utf-8') as fr:
        lines = fr.readlines()
    sent_, tag_ = [], []
    for line in lines:
        if line != '\n':
            [char, label] = line.strip().split()
            sent_.append(char)
            tag_.append(label)
        else:
            tag_ids = [label2idx[label] if label in label2idx else 0 for label in tag_]
            datas.append(sent_)
            labels.append(tag_ids)
            sent_, tag_ = [], []
    labels = sequence.pad_sequences(labels, maxlen=args.maxlen,padding="post")#datas中文本在使用Bert tokeizer时进行pad补齐
    return datas, labels

def get_mrc_data(data_path):
    text = open(data_path,encoding = "utf8").readlines()
    text = [each.split("\t") for each in text]
    sentence_pairs = []
    ans = []

    for each in text:
        if len(each)==3:
            context = each[0]
            que = each[1]
            sentence_pairs.append([context,que])
            ans.append(each[2])

    return sentence_pairs,ans

def get_pretrain_data(args):
    text = open(args.pretrain_data_path,encoding = "utf8").readlines()

    sentence_pairs = []
    text = clean(text)
    token_dict = {}
    with codecs.open(args.vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
            
    tokenizer = OurTokenizer(token_dict)
    for each in text:
        tem = each.strip()
        tem = tokenizer.tokenize(tem)
        tem = tem[1:-1]
        sentence_pairs.append([tem,[""]])
    return sentence_pairs

def clean(text):#数据预处理
    cleaned_text = []
    #数据清理
    for string in text:
        string = Traditional2Simplified(string)
        #string = re.sub("\?展开全文c","......",string)
        #string = re.sub("？[？]+","??",string)
        #string = re.sub("\?[\?]+","??",string)
        cleaned_text.append(string)
    return cleaned_text
#学习借鉴：https://spaces.ac.cn/archives/6736
#Tokenizer自带的_tokenize会自动去掉空格，然后有些字符会粘在一块输出，导致tokenize之后的列表不等于原来字符串的长度了，这样如果做序列标注的任务会很麻烦。

class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]') # 剩余的字符是[UNK]
        return R

def get_bert_input_cls(sentences,dict_path,max_len=128):#sentences是一个输入句子组成的一级列表["你是人","你不是人"]
    #得到字典
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
            
    tokenizer = OurTokenizer(token_dict)
    processed_sentences=[]
    x1=[]
    x2=[]
    #X1是经过编码后的集合，X2表示第一句和第二句的位置，记录的是位置信息
    #print(sentences)
    for sentence in sentences:
        #processed_sentences.append(tokenizer.tokenize(sentence))  
        indices, segments = tokenizer.encode(first=sentence, max_len=max_len)
        x1.append(indices)
        x2.append(segments)
    #x1 = sequence.pad_sequences(x1, maxlen=max_len)
    #x1 = sequence.pad_sequences(x2, maxlen=max_len)
    return x1,x2

def get_bert_input_ner(sentences,args):#sentences是一个输入句子组成的一级列表["你是人","你不是人"]
    #得到字典
    token_dict = {}
    with codecs.open(args.vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
            
    tokenizer = OurTokenizer(token_dict)
    processed_sentences=[]
    x1=[]
    x2=[]
    #X1是经过编码后的集合，X2表示第一句和第二句的位置，记录的是位置信息
    for sentence in sentences:  
        indices, segments = tokenizer.encode(first=sentence, max_len=args.maxlen)
        x1.append(indices)
        x2.append(segments)
    return x1,x2

def get_bert_input_mrc(sentences,answers,dict_path,max_len=128):#sentences是一个输入句子组成的一级列表["你是人","你不是人"]
    #得到字典
    token_dict = {}
    with codecs.open(dict_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
    tokenizer = OurTokenizer(token_dict)
    x1=[]
    x2=[]
    start_pointers = []
    end_pointers = []
    for sentence_pair,ans in zip(sentences,answers):
        try:
            #processed_sentences.append(tokenizer.tokenize(sentence))
            indices, segments = tokenizer.encode(first=sentence_pair[0],second = sentence_pair[1] , max_len=max_len)
            '''
            print("indices",indices)
            print("segments",segments)
            '''
            tem = np.zeros(len(indices))
            start = sentence_pair[1].find(ans)+len(sentence_pair[0])+1
            end = start+len(ans)
            tem[end] = 1
            end_pointers.append(np.expand_dims(tem, axis=-1))
            tem[end] = 0
            tem[start] = 1
            start_pointers.append(np.expand_dims(tem, axis=-1))
            x1.append(indices)
            x2.append(segments)
        except:
            pass#去掉长度太长的句子问题
    return x1,x2,start_pointers,end_pointers