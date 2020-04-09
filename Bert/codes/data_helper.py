import pandas as pd
import codecs
import re

from langconv import *#繁体字转化为简体字
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, extract_embeddings


def get_finetune_data(data_path):
    raw_data=open(data_path,encoding='utf8').read().split('\n')
    label=[i.split("\t")[0] for i in raw_data][0:-1] #最后一个为空行
    processed_test_label = []
    for each in label:
        if each == "-1":
            processed_test_label.append("2")
        else:
            processed_test_label.append(each)
    label=pd.get_dummies(pd.DataFrame(processed_test_label)) #转化为one-hot标签
    
    corpus=[i[2:].strip() for i in raw_data][0:-1]#最后一个为空行

    return corpus,label
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
        #print("之前:",string)
        try:
            string = Traditional2Simplified(string)
        except:
            pass
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
def get_bert_input(sentences,dict_path,max_len=128):#sentences是一个输入句子组成的一级列表["你是人","你不是人"]
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
