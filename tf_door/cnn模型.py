#-*-encoding:utf-8-*-
import numpy as np
from tensorflow.python.keras.preprocessing import image
from tensorflow.python.keras.applications import resnet50
import warnings

warnings.filterwarnings("ignore")


img = image.load_img('dog.png')
# print(image.img_to_array(img).shape)
model = resnet50.ResNet50(weights='imagenet')

img = image.load_img('dog.png', target_size=(224, 224))
img = image.img_to_array(img)
img = np.expand_dims(img, axis=0)
print(img.shape)

img = resnet50.preprocess_input(img)

#模型预测
pred = model.predict(img)
n = 10
top_n = resnet50.decode_predictions(pred, n)
for i in top_n[0]:
    print(i)