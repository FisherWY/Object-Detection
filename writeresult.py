import csv
import config

# 将检测结果写入csv文件中

# 结果存放文件夹目录
result_folder = config.RESULT_CSV_FOLDER

# 结果csv文件名
result_file = config.RESULT_CSV_NAME

def writeFile(result_folder, result_file, result):
    result_path = result_folder + result_file
    with open(result_path, mode='w', encoding='utf-8') as file:
        w = csv.writer(file)
        w.writerow(["ImageId","PredictionString"])
        for key,value in result.items():
            print("正在写入{}".format(key))
            w.writerow([key, value])
    file.close()
    print("写入结果完成")

if __name__ == '__main__':
    res = dict()
    res["123"] = "meibont"
    res["321"] = "ubnuinrtb"
    writeFile(result_folder, result_file, res)

