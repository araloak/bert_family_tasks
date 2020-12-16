from keras_bert import load_trained_model_from_checkpoint, Tokenizer, extract_embeddings
import codecs
#Reference：https://spaces.ac.cn/archives/6736
model_path="../pretrained_model/chinese_L-12_H-768_A-12/"
config_path = model_path+'bert_config.json'
checkpoint_path = model_path+'bert_model.ckpt'
vocab_path = model_path+'vocab.txt'

#Original _tokenize() method ignores blank spaces, therefore certain tokens cannot be sperated, which troubles NER tasks because len(string)!=len(tokenized_string)
class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]') #  [unused1] represent space
            else:
                R.append('[UNK]') # Unknown tokens [UNK]
        return R

if __name__=="__main__":
    #load dictionary
    token_dict = {}
    with codecs.open(vocab_path, 'r', 'utf8') as reader:
        for line in reader:
            token = line.strip()
            token_dict[token] = len(token_dict)

    tokenizer = OurTokenizer(token_dict)
    tem=tokenizer.tokenize(u'现在的年轻人')
    #tem: ['[CLS]', u'现', u'在', u'的', u'年', u'轻', u'人', '[SEP]']

    #embeddings = np.array(extract_embeddings(model_path, ['现在的年轻人']))
    #embeddings: a embedding matrix of [1,8,768]