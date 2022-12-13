# Pythonスクリプト
import struct
import numpy as np
import matplotlib.pyplot as plt

# バイナリデータを読み込む
with open('images/train-images-idx3-ubyte','rb') as f:
    # バイナリデータの最初の4バイトを取得
    magic_number = f.read(4)
    # バイトをint型に変換
    magic_number = struct.unpack('>i', magic_number)
    # バイナリデータの5バイト目から取得
    num_images = f.read(4)
    # バイトをint型に変換
    num_images = struct.unpack('>i', num_images)
    # 画像1枚あたりのバイト数を取得
    num_bytes = 28 * 28
    # 画像を格納する配列を定義
    images = []

    # バイナリデータから画像を一つ一つ取得
    for i in range(num_images[0]):
        # 画像1枚分のバイナリデータを取得
        image = f.read(num_bytes)
        # バイトをint型に変換
        image = struct.unpack('>784B',image)
        # 画像を配列に格納
        images.append(image)

    # 配列から画像を取得
    for index, image in enumerate(images):
        # 配列を28x28の配列に変換
        image = np.array(image).reshape(28,28)
        # 画像を保存
        plt.imsave('image_{}.png'.format(index), image, cmap='gray')