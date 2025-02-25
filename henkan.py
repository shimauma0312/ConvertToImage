import struct
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

def process_data(is_image_to_binary=True, input_path=None, output_path=None):
    """
    画像とバイナリ相互変換
    
    Parameters:
    is_image_to_binary (bool): True=画像→バイナリ変換, False=バイナリ→画像変換
    input_path (str): 入力ファイルまたはディレクトリのパス
    output_path (str): 出力先のパスまたはファイル名
    """
    
    # 出力先ディレクトリ
    if output_path and not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 画像→バイナリ
    if is_image_to_binary:
        convert_image_to_binary(input_path, output_path)
    # バイナリ→画像
    else:
        convert_binary_to_image(input_path, output_path)

def convert_image_to_binary(input_path, output_path):
    
    if not os.path.exists(input_path):
        print(f"エラー: 入力パス {input_path} が存在しません")
        return
    
    # 単一画像の場合
    if os.path.isfile(input_path):
        images = [np.array(Image.open(input_path).convert('L'))]
        file_names = [os.path.basename(input_path)]
    # ディレクトリ内の全画像を処理する場合
    else:
        images = []
        file_names = []
        for file in os.listdir(input_path):
            if file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                try:
                    img_path = os.path.join(input_path, file)
                    img = np.array(Image.open(img_path).convert('L'))
                    # 必要に応じてリサイズ（MNIST形式の場合は28x28）
                    if img.shape != (28, 28):
                        img = np.array(Image.open(img_path).convert('L').resize((28, 28)))
                    images.append(img)
                    file_names.append(file)
                except Exception as e:
                    print(f"画像 {file} の読み込みに失敗しました: {e}")
    
    if not images:
        print("変換する画像が見つかりませんでした")
        return
    
    # バイナリファイルの作成
    output_file = output_path if output_path else 'output.idx3-ubyte'
    
    with open(output_file, 'wb') as f:
        # マジックナンバー (2051 = 0x0803)
        f.write(struct.pack('>i', 2051))
        # 画像数
        f.write(struct.pack('>i', len(images)))
        # 行数
        f.write(struct.pack('>i', 28))
        # 列数
        f.write(struct.pack('>i', 28))
        
        # 各画像のデータを書き込み
        for img in images:
            flattened = img.flatten()
            f.write(struct.pack(f'>{len(flattened)}B', *flattened))
    
    print(f"{len(images)}枚の画像を {output_file} に変換しました")
    
    # 変換した画像の一覧を保存
    with open(f"{os.path.splitext(output_file)[0]}_info.txt", 'w', encoding='utf-8') as f:
        for i, name in enumerate(file_names):
            f.write(f"{i}: {name}\n")

def convert_binary_to_image(input_path, output_dir):
    """バイナリデータから画像を生成する関数"""
    
    if not os.path.exists(input_path):
        print(f"エラー: 入力ファイル {input_path} が存在しません")
        return
    
    # 出力ディレクトリが指定されていない場合はデフォルト設定
    if not output_dir:
        output_dir = 'output_images'
    
    # 出力ディレクトリの作成
    os.makedirs(output_dir, exist_ok=True)
    
    with open(input_path, 'rb') as f:
        # マジックナンバーの読み込み
        magic_number = struct.unpack('>i', f.read(4))[0]
        
        # マジックナンバーの確認（2051 = 0x0803 は画像データを示す）
        if magic_number != 2051:
            print(f"警告: マジックナンバーが画像データと一致しません: {magic_number}")
        
        # 画像数、行数、列数の読み込み
        num_images = struct.unpack('>i', f.read(4))[0]
        num_rows = struct.unpack('>i', f.read(4))[0]
        num_cols = struct.unpack('>i', f.read(4))[0]
        
        print(f"バイナリデータの情報: {num_images}枚の画像 ({num_rows}x{num_cols})")
        
        # 各画像のバイト数
        num_bytes = num_rows * num_cols
        
        # 画像の抽出
        for i in range(num_images):
            # 画像1枚分のバイナリデータを取得
            image_bytes = f.read(num_bytes)
            
            # バイトをint型に変換
            image_data = struct.unpack(f'>{num_bytes}B', image_bytes)
            
            # 配列を適切な次元に変換
            image = np.array(image_data).reshape(num_rows, num_cols)
            
            # 画像の保存
            output_path = os.path.join(output_dir, f'image_{i:05d}.png')
            plt.imsave(output_path, image, cmap='gray')
            
            # 進捗表示（大量の画像がある場合）
            if (i+1) % 1000 == 0 or i+1 == num_images:
                print(f"{i+1}/{num_images} 枚の画像を処理しました")
    
    print(f"バイナリデータから {num_images}枚の画像を {output_dir} ディレクトリに保存しました")

# ps
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='画像とバイナリデータを相互変換するツール')
    parser.add_argument('--mode', type=str, choices=['i2b', 'b2i'], required=True, 
                        help='処理モード: i2b=画像→バイナリ, b2i=バイナリ→画像')
    parser.add_argument('--input', type=str, required=True, 
                        help='入力ファイルまたはディレクトリのパス')
    parser.add_argument('--output', type=str, default=None, 
                        help='出力先のパス')
    
    args = parser.parse_args()
    
    # bool変数で処理モードを切り替え
    is_image_to_binary = (args.mode == 'i2b')
    
    # 処理実行
    process_data(is_image_to_binary, args.input, args.output)
