from scipy.stats import norm

# 指定正态分布的均值和标准差
mean = 0
std_dev = 1

K = 2

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