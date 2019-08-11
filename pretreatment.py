import pandas,os,numpy,boto3
from PIL import Image
from botocore import UNSIGNED
from botocore.config import Config
import config

# 数据集路径
csvPath = config.TRAIN_CSV_PATH

# 图片文件夹
imgPath = config.TRAIN_IMG_PATH

# 图片下载器
# s3 = boto3.resource('s3', config=Config(signature_version=UNSIGNED))

# 预处理单张图片模块
# 输入：文件路径
# 输出：处理结果
def readImgbyName(img_path, img_name):
    if os.path.exists(img_path):
        pic = Image.open(img_path)
        model = numpy.array(pic)
        return model
    else:
        # filename = 'train/' + img_name
        # print("{} is downloading to {}".format(filename, img_path))
        # s3.Bucket(config.AWS_BUCKET_NAME).download_file(filename, img_path)
        return None


# 图片及其特征数据预处理
# 输入：csv文件路径，图片文件夹路径，分块载入的开始位置，分块载入读取数量
# 输出：图片csv数据，图片numpy预处理后的数据
def prefun(csv_file, img_folder, start, row):
    # 图片的ID，标签，框框等位置信息
    img_csv = []
    # 图片经过numpy预处理之后的信息
    img_numpy = []

    # csv文件读取迭代器
    csv_reader = pandas.read_csv(csv_file, header=None, skiprows=start+1, chunksize=row)
    # 读取到的数据集
    csv_data = csv_reader.get_chunk()
    # 使用循环逐行预处理图片
    for i in range(row):
        # 生成图片路径
        img_path = img_folder + csv_data[0].values[i] + ".jpg"
        img_name = csv_data[0].values[i] + ".jpg"
        # 读取图片信息，如不存在返回None
        img_data = readImgbyName(img_path, img_name)
        # 图片不存在
        if img_data == None:
            print("{}.jpg 找不到，跳过".format(csv_data[0].values[i]))
            continue
        # 图片存在
        else:
            # 添加数据
            img_csv.append(csv_data.values[i])
            img_numpy.append(img_data)
            print("{}.jpg 预处理完成".format(csv_data[0].values[i]))
    # 关闭迭代器
    csv_reader.close()
    return img_csv, img_numpy

if __name__ == '__main__':
    csvData, imgData = prefun(csvPath, imgPath, 0, config.DEFAULT_READ_ROW)