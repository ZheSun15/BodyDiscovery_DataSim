import numpy as np

# 创建两个示例的二维 NumPy 数组
array1 = np.array([[1, 2, 3], [4, 5, 6]])
print(array1.shape)
array2 = np.array([[7, 8, 9], [10, 11, 12]])

# 将多个二维数组转化为字符串
array1_str = np.array2string(array1, separator=', ')
array2_str = np.array2string(array2, separator=', ')

# 打印转化后的字符串
print(array1_str + '\n' + array2_str)
a = (array1_str + '\n' + array2_str).replace('[', '').replace(']', '').replace(' ', '')
print(a)
