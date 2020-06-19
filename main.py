import os
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import time
import random
import pandas as pd


"""
# 使用GPU训练时，建议取消此注释
gpu = tf.config.experimental.list_physical_devices(device_type='GPU')
assert len(gpu) == 1
tf.config.experimental.set_memory_growth(gpu[0], True)
"""

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
CHAR_SET = number + alphabet
CHAR_SET_LEN = len(CHAR_SET)
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 120
MAX_CAPTCHA = 4
TRAIN_IMAGES_PATH = r'train'
TEST_IMAGES_PATH = r'test'

# 图片转为灰度图，如果是3通道图则计算，单通道图则直接返回
def rgb2gray(img):
    if len(img.shape) > 2:
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray / 255
    else:
        return img

# 对标签OneHot编码
def text2vec(text):
    if len(text) > MAX_CAPTCHA:
        raise ValueError('验证码最长{}个字符'.format(MAX_CAPTCHA))

    vector = np.zeros([MAX_CAPTCHA, CHAR_SET_LEN])
    for i, c in enumerate(text):
        idx = CHAR_SET.index(c)
        vector[i][idx] = 1.0
    return vector

# 将向量转为文本
def vec2text(vec):
    max_i = tf.argmax(vec, axis=1)
    text = []
    for i in max_i:
        text.append(CHAR_SET[i])
    return "".join(text)


"""
返回一个验证码的array形式和对应的字符串标签
:return: tuple (str, numpy.array)
"""
def get_captcha_text_image(img_path, img_name):
    # 标签
    label = img_name.split(".")[0]
    # 文件
    img_file = os.path.join(img_path, img_name)
    # print('%s' % img_file)
    captcha_image = Image.open(img_file)
    captcha_array = np.array(captcha_image)  # 向量化
    return label, captcha_array


def load_data(batch_size=None, images_path=TRAIN_IMAGES_PATH):
    images_list = os.listdir(images_path)
    # 打乱顺序
    random.seed(time.time())
    random.shuffle(images_list)

    if batch_size is None:
        images_total = int(len(images_list))
    else:
        images_total = batch_size
        images_list = images_list[:images_total]

    x = np.zeros([images_total, IMAGE_HEIGHT, IMAGE_WIDTH, 1])
    y = np.zeros([images_total, MAX_CAPTCHA, CHAR_SET_LEN])

    for i, image_name in enumerate(images_list):
        label, image_array = get_captcha_text_image(images_path, image_name)
        image_array = rgb2gray(image_array)  # 灰度化图片并归一化
        x[i, :] = tf.reshape(image_array, (IMAGE_HEIGHT, IMAGE_WIDTH, 1))
        y[i, :] = text2vec(label)
    return x, y


def cnn():
    model = models.Sequential()
    # 第一层卷积核，卷积核大小为3*3，32个，颜色通道为1
    model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='valid', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, 1)))
    model.add(layers.BatchNormalization())
    model.add(layers.ReLU())
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))
    # 第二层卷积核，卷积核大小为3*3，64个
    model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same'))
    model.add(layers.ReLU())
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # 第三层卷积核，卷积核大小为3*3，128个
    model.add(layers.Conv2D(128, kernel_size=(3, 3),  padding='same'))
    model.add(layers.ReLU())
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))
    # 第四层卷积核，卷积核大小为3*3，256个
    model.add(layers.Conv2D(256, kernel_size=(3, 3), padding='same'))
    model.add(layers.ReLU())
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(layers.Flatten())
    model.add(layers.Dense(1024))
    model.add(layers.ReLU())
    model.add(layers.Dropout(0.4))

    model.add(layers.Dense(MAX_CAPTCHA * CHAR_SET_LEN))
    model.add(layers.Reshape([MAX_CAPTCHA, CHAR_SET_LEN]))
    model.add(layers.Softmax())

    return model


def train():
    x, y = load_data()

    if os.path.exists('simple1.h5'):
        # 加载训练好的模型
        model = models.load_model('simple1.h5')
    else:
        model = cnn()
    model.compile(optimizer='adam',
                  metrics=['accuracy'],
                  loss='categorical_crossentropy')
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', patience=10, mode='auto')
    h = model.fit(x, y, batch_size=256, epochs=150, validation_split=0.1, validation_freq=1, callbacks=[reduce_lr])
    model.save('simple1.h5')

    return h

def evaluate():
    if os.path.exists('simple1.h5'):
        # 加载训练好的模型
        model = models.load_model('simple1.h5')
    else:
        raise Exception('请先训练模型！！！')

    (x, y) = load_data(images_path=TEST_IMAGES_PATH)
    print('x shape: {}\ny shape: {}'.format(x.shape, y.shape))
    test_loss, test_acc = model.evaluate(x, y)
    print('loss: {} >>>>>>> accuracy: {}'.format(test_loss, test_acc))


def loss_acc_curve(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()

def predict():
    if os.path.exists('simple1.h5'):
        # 加载训练好的模型
        model = models.load_model('simple1.h5')
    else:
        raise Exception('请先训练模型！！！')

    (x, y) = load_data(images_path=TEST_IMAGES_PATH)
    print('x shape: {}\ny shape: {}'.format(x.shape, y.shape))
    random_list = list()
    count = 0
    while True:
        num = random.randint(0, len(x))
        if num not in random_list:
            random_list.append(num)
            count += 1
        if count > 9:
            break

    y_ = model.predict(x)
    for i in random_list:
        print('Predictive value: {}  \nActual value: {}'.format(y_[i], y[i]))
        label = vec2text(y[i])
        img_name = label + '.png'
        img_file = os.path.join(TEST_IMAGES_PATH, img_name)
        img = Image.open(img_file)
        plt.figure(label)
        plt.figure(num=1, figsize=(2, 1), )
        plt.title('Predictive value: {}  >>>>   Actual value: {}'.format(vec2text(y_[i]), label))
        plt.axis('off')  # 不显示坐标轴
        plt.imshow(img)
        plt.show()


if __name__ == '__main__':
    his = train()
    loss_acc_curve(his)
    evaluate()
    predict()
