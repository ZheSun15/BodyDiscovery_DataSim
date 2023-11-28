import numpy as np
import random

# # 创建两个示例的二维 NumPy 数组
# array1 = np.array([[1, 2, 3], [4, 5, 6]])
# print(array1.shape)
# array2 = np.array([[7, 8, 9], [10, 11, 12]])

# # 将多个二维数组转化为字符串
# array1_str = np.array2string(array1, separator=', ')
# array2_str = np.array2string(array2, separator=', ')

# # 打印转化后的字符串
# print(array1_str + '\n' + array2_str)
# a = (array1_str + '\n' + array2_str).replace('[', '').replace(']', '').replace(' ', '')
# print(a)



# def split_range(n, p):
#     # 计算每份的长度
#     length = n // p
    
#     # 计算余数，用于处理不能整除的情况
#     remainder = n % p
    
#     # 生成分组
#     groups = [length] * (p - remainder) + [length + 1] * remainder

#     random.shuffle(groups)
#     APIlist = []
#     for q in range(p):
#         APIlist += [q] * groups[q]
    
#     random.shuffle(APIlist)

#     return APIlist


# # 示例：把范围 1 到 10 均匀分成 3 份
# result = split_range(10, 3)
# print(result)




# def API_list_generate(Q, T):
#     nqs = int(T/Q)
#     APIlist = []
#     for q in range(Q):
#         APIlist += [q] * nqs
#     APIlist_tail = np.random.choice(a=Q, size=(T-nqs*Q), replace=False)
#     APIlist += APIlist_tail.tolist()
#     APIlist = random.shuffle(APIlist)

#     return APIlist

# states = API_list_generate(10,109)
# print(states) 



import seaborn as sns
import matplotlib.pyplot as plt
fmri = sns.load_dataset("fmri")
sns.relplot(
    data=fmri, kind="line",
    x="timepoint", y="signal", col="region",
    hue="event", style="event",
)
plt.show()
pass