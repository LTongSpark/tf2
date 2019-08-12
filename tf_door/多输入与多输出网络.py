#-*-encoding:utf-8-*-
import numpy as np
import tensorflow as tf
from tensorflow.python.keras import layers

'''
构建一个根据文档内容，标签和标题，预测文档优先级和执行部门的网络
超参
'''
num_words =2000
num_tags =12
num_departments = 4

'''输入'''
body_input = tf.keras.layers.Input(shape = (None,) ,name ='body')
title_input = tf.keras.layers.Input(shape = (None,) ,name ='title')
tag_input = tf.keras.layers.Input(shape = (num_tags,) ,name ='tag')

'''嵌入层'''
body_feat = layers.Embedding(num_words ,64)(body_input)
title_feat = layers.Embedding(num_words ,64)(title_input)

'''特征提取层'''

body_feat = layers.LSTM(32)(body_feat)
title_feat = layers.LSTM(32)(title_feat)
features = layers.concatenate([title_feat,body_feat, tag_input])
print(tag_input ,body_feat ,title_feat ,features)

''' 分类层'''
priority_pred = layers.Dense(1, activation='sigmoid', name='priority')(features)
department_pred = layers.Dense(num_departments, activation='softmax', name='department')(features)
print(priority_pred ,department_pred)
'''构建模型'''
model = tf.keras.Model(inputs=[body_input, title_input, tag_input],
                    outputs=[priority_pred, department_pred])

model.summary()
tf.keras.utils.plot_model(model, 'multi_model.png', show_shapes=True)

model.compile(optimizer=tf.keras.optimizers.RMSprop(1e-3),
             loss={'priority': 'binary_crossentropy',
                  'department': 'categorical_crossentropy'},
             loss_weights=[1., 0.2] ,
              metrics=['accuracy'])

import numpy as np
# 载入输入数据
title_data = np.random.randint(num_words, size=(1280, 10))
body_data = np.random.randint(num_words, size=(1280, 100))
tag_data = np.random.randint(2, size=(1280, num_tags)).astype('float32')
# 标签
priority_label = np.random.random(size=(1280, 1))
department_label = np.random.randint(2, size=(1280, num_departments))
# 训练
history = model.fit(
    {'title': title_data, 'body':body_data, 'tag':tag_data},
    {'priority':priority_label, 'department':department_label},
    batch_size=32,
    epochs=5
)

print()

# 查看训练过程
import matplotlib.pyplot as plt

def plot_graphs(history, string):
    plt.plot(history.history[string])
    #plt.plot(history.history['val_'+string])
    plt.xlabel('epochs')
    plt.ylabel(string)
    #plt.legend([string, 'val_'+string])
    plt.show()

plot_graphs(history, 'department_accuracy')














