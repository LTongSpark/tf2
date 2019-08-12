#-*-encoding:utf-8-*-
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers

train_x = np.random.random(size=(1000,72))
train_y = np.random.random(size=(1000 ,10))

val_x = np.random.random(size =(200,72))
val_y = np.random.random(size=(200 ,10))

#构建模型
modle = tf.keras.Sequential()
modle.add(layers.Dense(32 ,activation='relu'))
modle.add(layers.Dense(32,activation='relu'))
modle.add(layers.Dense(10 ,activation=tf.sigmoid))

modle.compile(optimizer=tf.keras.optimizers.Adam(0.01),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

#输入numpy的数据
modle.fit(x = train_x, y = train_y, batch_size =100 ,epochs=10,validation_data=(val_x,val_y))

#输入dataset中的数据

dataset = tf.data.Dataset.from_tensor_slices((train_x ,train_y))
dataset = dataset.batch( 32)
dataset = dataset.repeat()

val_dataset = tf.data.Dataset.from_tensor_slices(val_x,val_y)
val_dataset = val_dataset.batch(32)
val_dataset = val_dataset.repeat()

modle.fit(dataset,epochs=10 ,steps_per_epoch=30 ,validation_data=val_dataset ,validation_steps=3)

'''
评估和预测
'''

test_x = np.random.random(size=(1000,72))
test_y = np.random.random(size=(200 ,10))
modle.evaluate(test_x, test_y ,batch_size=32)

test_data = tf.data.Dataset.from_tensor_slices((test_x,test_y))
test_data.batch(32).repeat()

modle.evaluate(test_data, batch_size = 32)

#预测
result = modle.predict(test_x,batch_size=32)
print(result)
