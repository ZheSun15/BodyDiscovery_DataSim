from scipy.stats import norm

# 指定正态分布的均值和标准差
mean = 0
std_dev = 1

K = 6

# alpha = [0.05, 0.01]
# ppf_param = 1 - (alpha/2/K)

# 计算累积分布函数（CDF）的值，即百分位数
# 例如，计算正态分布的第25、50和75百分位数
# percentile_25 = norm.ppf(0.25, loc=mean, scale=std_dev)
# percentile_50 = norm.ppf(0.5, loc=mean, scale=std_dev)
# percentile_alpha = norm.ppf(ppf_param, loc=mean, scale=std_dev)
percentile_005 = norm.ppf(1 - (0.05/2/K), loc=mean, scale=std_dev)
percentile_001 = norm.ppf(1 - (0.01/2/K), loc=mean, scale=std_dev)

# print("25th Percentile:", percentile_25)
# print("50th Percentile (Median):", percentile_50)
# print("75th Percentile:", percentile_75)
print(percentile_005)
print(percentile_001)

# import os
# import numpy as np
# import glob
# import json
# import re

# # 自定义排序键函数，提取文件名中的数字并将其转换为整数
# def custom_sort_key(filename):
#     # 使用正则表达式提取文件名中的数字部分
#     match = re.search(r'T\d+', filename)
#     if match:
#         number_str = match.group().replace('T', '')
#         # # 将数字部分转换为带有固定宽度的字符串，例如 "00010"
#         # number_str = number_str.zfill(5)  # 假设宽度为 5
#         return int(number_str)  # 将提取的数字部分转换为整数
#     else:
#         return filename  # 如果没有数字，则按原始文件名排序


# prediction_root = "./prediction/video_S3_prediction"
# save_root = "./prediction/xiaoxiao/S3"
# # shape: (T_num, N_num)

# if not os.path.exists(save_root):
#     os.makedirs(save_root)

# result_pths = sorted(glob.glob(prediction_root + "/*_round*_Q10_N20_T*_plist.json"))
# # 使用自定义排序键函数进行排序
# sorted_filenames = sorted(result_pths, key=custom_sort_key)

# min_ps = np.zeros((len(sorted_filenames), 20))

# for id, result_pth in enumerate(sorted_filenames):
#     # load json file
#     pred = json.load(open(result_pth, "r"))
#     array = np.array(pred)

#     _min_ps = np.min(array, axis=0)
#     _min_ps = np.min(_min_ps, axis=-1)

#     min_ps[id, :] = _min_ps

# # save json
# save_pth = os.path.join(save_root, os.path.basename(result_pth))
# json.dump(min_ps.tolist(), open(save_pth, "w"), indent=4)