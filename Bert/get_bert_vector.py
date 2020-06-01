from keras_bert import load_trained_model_from_checkpoint, Tokenizer, extract_embeddings
import codecs
import sys
import numpy as np
#学习借鉴：https://spaces.ac.cn/archives/6736
model_path="../pretrained_model/chinese_L-12_H-768_A-12/"

config_path = model_path+'bert_config.json'
checkpoint_path = model_path+'bert_model.ckpt'
vocab_path = model_path+'vocab.txt'

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
if __name__=="__main__":
    #得到字典
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)
            

    tokenizer = OurTokenizer(token_dict)
    tem=tokenizer.tokenize(u'现在的年轻人')
    #print(tem)
    # 输出是 ['[CLS]', u'现', u'在', u'的', u'年', u'轻', u'人', '[SEP]']
    #直接得到[1,8,768]维的词向量矩阵
    embeddings = np.array(extract_embeddings(model_path, ['现在的年轻人']))
