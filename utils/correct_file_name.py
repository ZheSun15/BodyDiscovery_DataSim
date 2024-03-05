import os
import re

# 获取目录中的所有文件
directory = './prediction/ablation_new_240227/noises'  # 你的目录
filenames = os.listdir(directory)

# 对每个文件进行处理
for filename in filenames:
    # 使用正则表达式找到文件名中的浮点数
    match = re.search(r'\d+\.\d+', filename)
    if match:
        # 获取浮点数
        number = float(match.group())
        # 获取小数部分
        decimal_part = match.group().split('.')[1]

        # 如果小数部分的位数大于等于2，那么进行四舍五入和重命名
        if len(decimal_part) >= 2:
            rounded_number = round(number, 1)

            # 生成新的文件名
            new_filename = filename.replace(str(number), str(rounded_number))

            # 重命名文件
            os.rename(os.path.join(directory, filename), os.path.join(directory, new_filename))
            print("\nRename: \n    " + filename)