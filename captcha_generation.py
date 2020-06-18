#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import random
from captcha.image import ImageCaptcha
import os
from PIL import Image
import shutil


# 验证码的存储路径
ORIGIN_IMG_PATH = r'origin'
TRAIN_IMG_PATH = r'train'
TEST_IMG_PATH = r'test'


number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']
CHAR_SET = number + alphabet
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 120

"""
功能：生成验证码
参数：验证码包含的字符数目，原始数据集的规模
返回：
"""
def gen_captcha(length, scale):
    if not os.path.exists(ORIGIN_IMG_PATH):
        os.makedirs(ORIGIN_IMG_PATH)

    for i in range(0, scale):
        # 生成4位随机字符串
        random_str = ''.join(random.sample(CHAR_SET, length))
        generator = ImageCaptcha(width=IMAGE_WIDTH, height=IMAGE_HEIGHT)
        # 生成图片
        img = generator.generate_image(random_str)
        path = os.path.join(ORIGIN_IMG_PATH, "{}.{}".format(random_str, 'png'))
        img.save(path)

        if i % 500 == 0:
            print("Creating %d/%d" % (i, scale))


"""
功能：分离训练集和测试集
参数：
返回：
"""
def split():
    if not os.path.exists(ORIGIN_IMG_PATH):
        print("【警告】找不到目录{}，即将创建".format(ORIGIN_IMG_PATH))
        os.makedirs(ORIGIN_IMG_PATH)

    print("开始分离原始图片集为：测试集（5%）和训练集（95%）")

    # 图片名称列表和数量
    img_list = os.listdir(ORIGIN_IMG_PATH)
    total_count = len(img_list)
    print("共分配{}张图片到训练集和测试集".format(total_count))

    # 创建文件夹
    if not os.path.exists(TRAIN_IMG_PATH):
        os.mkdir(TRAIN_IMG_PATH)

    if not os.path.exists(TEST_IMG_PATH):
        os.mkdir(TEST_IMG_PATH)

    # 测试集
    test_count = int(total_count*0.05)
    test_set = set()
    for i in range(test_count):
        while True:
            file_name = random.choice(img_list)
            if file_name in test_set:
                pass
            else:
                test_set.add(file_name)
                img_list.remove(file_name)
                break

    test_list = list(test_set)
    print("测试集数量为：{}".format(len(test_list)))
    for file_name in test_list:
        src = os.path.join(ORIGIN_IMG_PATH, file_name)
        dst = os.path.join(TEST_IMG_PATH, file_name)
        shutil.move(src, dst)

    # 训练集
    train_list = img_list
    print("训练集数量为：{}".format(len(train_list)))
    for file_name in train_list:
        src = os.path.join(ORIGIN_IMG_PATH, file_name)
        dst = os.path.join(TRAIN_IMG_PATH, file_name)
        shutil.move(src, dst)

    if os.listdir(ORIGIN_IMG_PATH) == 0:
        print("migration done")


if __name__ == '__main__':
    gen_captcha(4, 60000)
    split()
