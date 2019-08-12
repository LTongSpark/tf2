#-*-encoding:utf-8-*-
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers
'''
1.1  创建网络
'''

input = tf.keras.Input(shape = (784,))
h1 = layers.Dense(32 ,activation=tf.nn.relu)(input)
h2 = layers.Dense(32 ,activation=tf.nn.relu)(h1)

output =layers.Dense(10 ,activation=tf.nn.softmax)(h2)
model = tf.keras.Model(inputs = input ,outputs  = output ,name='mnist')
model.summary()
tf.keras.utils.plot_model(model,'mnist_model.png')
tf.keras.utils.plot_model(model, 'model_info.png', show_shapes=True)

'''
训练验证和测试
'''

(x_train ,y_train) ,(x_test ,y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000 ,784).astype(tf.float32)/255
x_test = x_test.reshape(10000 ,784).astype(tf.float32)/255
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
             loss='sparse_categorical_crossentropy', # 直接填api，后面会报错
             metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs =5 ,batch_size=65 ,validation_split=0.2)
test_score = model.evaluate(x_test, y_test ,verbose=0)
print(test_score)

print('test loss :' ,test_score[0])
print('test acc :' ,test_score[1])

model.save('model_save.h5')
del model
model = tf.keras.models.load_model('model_save.h5')




