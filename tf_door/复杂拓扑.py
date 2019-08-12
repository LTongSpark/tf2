#-*-encoding:utf-8-*-
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import layers

'''
tf.keras.Sequential 模型是层的简单堆叠，无法表示任意模型。使用 Keras 函数式 API 可以构建复杂的模型拓扑，例如：
多输入模型，
多输出模型，
具有共享层的模型（同一层被调用多次），
具有非序列数据流的模型（例如，残差连接）。
使用函数式 API 构建的模型具有以下特征：
层实例可调用并返回张量。
输入张量和输出张量用于定义 tf.keras.Model 实例。
此模型的训练方式和 Sequential 模型一样。
'''

train_x = np.random.random((1000, 72))
train_y = np.random.random((1000, 10))
val_x = np.random.random((200 ,72))
val_y = np.random.random((100 ,10))
input_x = tf.keras.Input(shape = [72,] ,dtype=tf.float64)
hidden1  = layers.Dense(32,activation='relu')(input_x)
hidden2 = layers.Dense(16 ,activation='relu')(hidden1)
pred = layers.Dense(10 ,activation=tf.sigmoid)(hidden2)

#构建模型
model = tf.keras.Model(inputs = input_x ,outputs=pred)
model.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.categorical_crossentropy ,
              metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=32, epochs=5)
print(input_x ,hidden1,hidden2 ,pred)

'''
模型子类化
通过对 tf.keras.Model 进行子类化并定义您自己的前向传播来构建完全可自定义的模型。
在 init 方法中创建层并将它们设置为类实例的属性。在 call 方法中定义前向传播
'''

class MyModel(tf.keras.Model):
    def __init__(self,num_classes):
        super(MyModel, self).__init__(name='my_model')
        self.num_classes = num_classes
        self.layers1 = layers.Dense(32,activation=tf.nn.relu)
        self.layers2 = layers.Dense(num_classes ,activation=tf.nn.sigmoid)
    def call(self, inputs):
        h1 = self.layers1(inputs)
        out = self.layers2(h1)
        return out
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.num_classes
        return tf.TensorShape(shape)

model  = MyModel(num_classes=10)
model.compile(optimizer=tf.keras.optimizers.Adam(0.01) ,
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

model.fit(train_x, train_y, batch_size =16,epochs=10)

'''
自定义层，通过对 tf.keras.layers.Layer 进行子类化并实现以下方法来创建自定义层
build：创建层的权重。使用 add_weight 方法添加权重。
call：定义前向传播。
compute_output_shape：指定在给定输入形状的情况下如何计算层的输出形状。
或者，可以通过实现 get_config 方法和 from_config 类方法序列化层。
'''

class MyLayer(layers.Layer):
    def __init__(self,out_dim,**kwargs):
        self.out_dim = out_dim
        super(MyLayer,self).__init__(* kwargs)

    def build(self, input_shape):
        shape = tf.TensorShape((input_shape[1] ,self.out_dim))
        self.kernel = self.add_weight(name ='kernel1' ,shape=shape ,initializer='uniform' ,trainable=True)
        super(MyLayer ,self).build(input_shape)

    def call(self, inputs):
        return tf.matmul(inputs, self.kernel)
    def compute_output_shape(self, input_shape):
        shape = tf.TensorShape(input_shape).as_list()
        shape[-1] = self.out_dim
        return tf.TensorShape(shape)

    def get_config(self):
        base_config = super(MyLayer,self).get_config()
        base_config['output_dim'] = self.out_dim
        return base_config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

model = tf.keras.Sequential(
    [
        MyLayer(10),
        layers.Activation(tf.nn.softmax)
    ]
)

model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.01) ,
              loss=tf.keras.losses.categorical_crossentropy ,
              metrics=['accuracy'])

model.fit(train_x, train_y, batch_size =16 ,epochs=10)

'''
回调
'''

callback = [
    tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3),
    tf.keras.callbacks.TensorBoard(log_dir="./logs")
]
model.fit(train_x, train_y, batch_size =16 ,epochs=10 ,callbacks=callback,validation_data=(val_x ,val_y))


'''
保存和恢复
1.1 权重保存
'''
model = tf.keras.Sequential([
    layers.Dense(64 ,activation = tf.nn.relu) ,
    layers.Dense(10 ,activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.01) ,
              loss=tf.keras.losses.categorical_crossentropy ,
              metrics=['accuracy'])

model.save_weights('./weights/model')
model.load_weights('./weights/model')
model.save_weights('./model.h5')
model.load_weights('./model.h5')

'''
1.2  保存网络结构
'''

#序列化成json
import json
import pprint
json_str = model.to_json()
pprint.pprint(json.loads(json_str))
fresh_model = tf.keras.models.model_from_json(json)

# 保持为yaml格式  #需要提前安装pyyaml

yaml_str = model.to_yaml()
print(yaml_str)
fresh_model = tf.keras.models.model_from_yaml(yaml_str)

'''
1,3  保存整个模型
'''

model = tf.keras.Sequential([
    layers.Dense(10 ,activation=tf.nn.softmax ,input_shape=(72,)),
    layers.Dense(10 ,activation=tf.nn.sigmoid)
])

model.compile(optimizer=tf.keras.optimizers.Adam(lr = 0.01) ,
              loss=tf.keras.losses.categorical_crossentropy ,
              metrics=['accuracy'])

model.fit(train_x ,train_y, batch_size =16 ,epochs=10)
model.save("all_model.h5")
model = tf.keras.models.save_model("all_model.h5")

'''
将keras用于estimator
Estimator API 用于针对分布式环境训练模型。它适用于一些行业使用场景，例如用大型数据集进行分布式训练并导出模型以用于生产
'''

estimator = tf.keras.estimator.model_to_estimator(model)























