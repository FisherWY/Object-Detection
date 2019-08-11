import os,shutil
import config

def mkdir(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.mkdir(folder_path)

if __name__ == '__main__':
    if os.path.exists(config.RESULT_CSV_FOLDER + config.RESULT_CSV_NAME):
        os.remove(config.RESULT_CSV_FOLDER + config.RESULT_CSV_NAME)
        print("发现目标文件，已删除")