#-*-encoding:utf-8-*-
import tensorflow as tf
from tensorflow.python.keras import layers
import warnings

warnings.filterwarnings("ignore")

'''
模型堆叠，最常见的为层的堆叠，tf.keras.Sequential模型
activation  设置层的激活函数
kernel_initializer和bias_initializer  创建层权重（核和偏差）的初始化方案
kernel_regularizer和bias_regularizer  应用层权重（核和偏差） 的正则化方案 ，例如l1 和l2
'''

model = tf.keras.Sequential()
model.add(layers.Dense(32 ,activation='relu'))
model.add(layers.Dense(10 ,activation=tf.sigmoid ,kernel_initializer=tf.keras.initializers.glorot_normal))
model.add(layers.Dense(10 ,activation=tf.sigmoid ,kernel_regularizer=tf.keras.regularizers.l2(0.01)))


'''
构建好模型后，通常调用compile方法配置改模型的学习流程
optimizer 优化梯度下降
loss 损失函数
metrics 
'''

model.compile(optimizer=tf.keras.optimizers.Adam(0.01) ,
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])




print(model)


