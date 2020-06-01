import tensorflow as tf
import keras

from keras.layers import *
from keras.models import Model
from keras import backend as K

from keras_bert.optimizers import AdamWarmup
from keras_bert import load_trained_model_from_checkpoint
from bert4keras.optimizers import Adam, extend_with_piecewise_linear_lr
from bert4keras.models import build_transformer_model


# 派生为带分段线性学习率的优化器。
# 其中name参数可选，但最好填入，以区分不同的派生优化器。
AdamLR = extend_with_piecewise_linear_lr(Adam, name='AdamLR')

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

#用于不均衡样本多分类
def multi_category_focal_loss2(gamma=2., alpha=.25):
    """
    focal loss for multi category of multi label problem
    适用于多分类或多标签问题的focal loss
    alpha控制真值y_true为1/0时的权重
        1的权重为alpha, 0的权重为1-alpha
    当你的模型欠拟合，学习存在困难时，可以尝试适用本函数作为loss
    当模型过于激进(无论何时总是倾向于预测出1),尝试将alpha调小
    当模型过于惰性(无论何时总是倾向于预测出0,或是某一个固定的常数,说明没有学到有效特征)
        尝试将alpha调大,鼓励模型进行预测出1。
    Usage:
     model.compile(loss=[multi_category_focal_loss2(alpha=0.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    epsilon = 1.e-7
    gamma = float(gamma)
    alpha = tf.constant(alpha, dtype=tf.float32)

    def multi_category_focal_loss2_fixed(y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        y_t = tf.multiply(y_true, y_pred) + tf.multiply(1 - y_true, 1 - y_pred)
        ce = -K.log(y_t)
        weight = tf.pow(tf.subtract(1., y_t), gamma)
        fl = tf.multiply(tf.multiply(weight, ce), alpha_t)
        loss = tf.reduce_mean(fl)
        return loss

    return multi_category_focal_loss2_fixed

#用于不均衡样本二分类
def focal_loss(gamma=2., alpha=.25):
	def focal_loss_fixed(y_true, y_pred):
		pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
		pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
		return -K.mean(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) - K.mean((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
	return focal_loss_fixed

#用于mrc
def build_mrc_bert(args,training= False):
    bert_model = load_trained_model_from_checkpoint(args.config_path, args.checkpoint_path, seq_len=None,training = training)  #加载预训练模型
    print(bert_model)
    for l in bert_model.layers:
        #if "-12-" in l.name or "-11-" in l.name or "-10-" in l.name:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])

    #x = bert_model.get_layer(name='Encoder-{}-FeedForward-Norm'.format(12))(x)
    #x = Lambda(lambda x: x, output_shape=lambda s:s)(x)
    
    p_start = Dense(1, activation='sigmoid',name = "p_start")(x)
    p_end = Dense(1, activation='sigmoid',name = "p_end")(x)

    model = Model([x1_in, x2_in],[ p_start,p_end])

    model.compile(loss=focal_loss(gamma=2., alpha=.25),
                  #loss='binary_crossentropy',
                  optimizer=Adam(args.lr),   #用足够小的学习率
                  #loss_weights=[1., 1.]
                  #metrics=['accuracy']
        )

    print(model.summary())
    return model

#用于ner
def build_ner_bert(args,training= False):
    bert_model = load_trained_model_from_checkpoint(args.config_path, args.checkpoint_path, seq_len=None,training = training)  #加载预训练模型

    for l in bert_model.layers:
        #if "-12-" in l.name or "-11-" in l.name or "-10-" in l.name:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    #x = Lambda(lambda x: x[:, 0])(x) # 取出[CLS]对应的向量用来做分类
    
    p = Dense(args.nclass, activation='softmax',name = "p")(x)

    model = Model([x1_in, x2_in], p)
    
    model.compile(
        #loss=multi_category_focal_loss2(gamma=2., alpha=.25),
        loss='categorical_crossentropy',
        optimizer=Adam(args.lr),
        #metrics=["accuracy"]
        )
    
    print(model.summary())
    return model    

#用于文本分类
def build_cls_bert(args):
    bert_model = load_trained_model_from_checkpoint(args.config_path, args.checkpoint_path, seq_len=None,use_adapter=True)  # 加载预训练模型

    for l in bert_model.layers:
        #if "-12-" in l.name or "-11-" in l.name or "-10-" in l.name:
        l.trainable = True#选择冻结的层

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    
    a1 = bert_model.get_layer(name='Encoder-{}-FeedForward-Norm'.format(12))(x)#获取最后一层输出
    a2 = bert_model.get_layer(name='Encoder-{}-FeedForward-Norm'.format(11))(x)#获取倒数第二层输出
    
    x = Add()([a1, a2])
    
    x = Lambda(lambda x: x[:, 0])(x) # 取出[CLS]对应的向量用来做分类
    
    p = Dense(args.nclass, activation='softmax')(x)
 
    model = Model([x1_in, x2_in], p)
    model.compile(loss = 'categorical_crossentropy',#多分类
                  #loss = focal_loss(gamma=2., alpha=.25),
                  #loss=multi_category_focal_loss2(gamma=2., alpha=.25),                  #optimizer=Adam(args.lr),    #用足够小的学习率
                  #optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1,2000: 0.1}),
                  optimizer=Adam(args.lr),
                  metrics=['accuracy',f1])
    print(model.summary())
    return model

def pretrain_bert(args,training= True):
    bert_model = load_trained_model_from_checkpoint(args.config_path, args.checkpoint_path, seq_len=None,training = training)  #加载预训练模型
    for l in bert_model.layers:
        l.trainable = True

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))
    x3_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in,x3_in])

    model = Model([x1_in, x2_in,x3_in], x)
    
    model.compile(
        optimizer=AdamWarmup(
            decay_steps=100000,
            warmup_steps=10000,
            learning_rate=1e-4,
            weight_decay=0.01,
            weight_decay_pattern=['embeddings', 'kernel', 'W1', 'W2', 'Wk', 'Wq', 'Wv', 'Wo'],
        ),
        loss=keras.losses.sparse_categorical_crossentropy,
    )
    print(model.summary())
    return model

def build_cls_albert(args):
    bert_model = build_transformer_model(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
         model='albert',
        # return_keras_model=False,
    )

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
    if args.nclass!=2:
        p = Dense(args.nclass, activation='softmax')(x)
        loss = 'categorical_crossentropy'
    else:
        p = Dense(1, activation='sigmoid')(x)
        loss = 'binary_crossentropy'
    model = Model([x1_in, x2_in], p)
    model.compile(loss=loss,  # 多分类
                  #optimizer=Adam(args.lr),  # 用足够小的学习率
                  optimizer=AdamLR(learning_rate=1e-4, lr_schedule={1000: 1, 2000: 0.1}),
                  metrics=['accuracy', f1])
    print(model.summary())
    return model

def build_ner_albert(args):
    bert_model = build_transformer_model(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
         model='albert',
        # return_keras_model=False,
    )

    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])
    #x = Lambda(lambda x: x[:, 0])(x) # 取出[CLS]对应的向量用来做分类
    
    p = Dense(args.nclass, activation='softmax',name = "p")(x)

    model = Model([x1_in, x2_in], p)
    
    model.compile(
        #loss=multi_category_focal_loss2(gamma=2., alpha=.25),
        loss='categorical_crossentropy',
        optimizer=Adam(args.lr),
        #metrics=["accuracy"]
        )
    
    print(model.summary())
    return model

def build_mrc_albert(args):
    bert_model = build_transformer_model(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
         model='albert',
        # return_keras_model=False,
    )
    x1_in = Input(shape=(None,))
    x2_in = Input(shape=(None,))

    x = bert_model([x1_in, x2_in])

    #x = bert_model.get_layer(name='Encoder-{}-FeedForward-Norm'.format(12))(x)
    #x = Lambda(lambda x: x, output_shape=lambda s:s)(x)
    
    p_start = Dense(1, activation='sigmoid',name = "p_start")(x)
    p_end = Dense(1, activation='sigmoid',name = "p_end")(x)

    model = Model([x1_in, x2_in],[ p_start,p_end])

    model.compile(loss=focal_loss(gamma=2., alpha=.25),
                  #loss='binary_crossentropy',
                  optimizer=Adam(args.lr),   #用足够小的学习率
                  #loss_weights=[1., 1.]
                  #metrics=['accuracy']
        )

    print(model.summary())
    return model