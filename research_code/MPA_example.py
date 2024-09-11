import numpy as np
import pywt

def matching_pursuit_algorithm(signal, dictionary, max_iterations):
    residual = signal.copy()
    coefficients = []

    for _ in range(max_iterations):
        # 选择字典中最匹配的基函数
        atom = pywt.waverec2([residual], dictionary, mode='per')

        # 调整形状以匹配
        atom = atom.flatten()

        # 获取当前系数
        current_coefficient = np.sum(residual * atom)

        # 避免分母为零
        denominator = np.sum(atom * atom)
        if denominator != 0:
            current_coefficient /= denominator

        # 更新残差
        residual = residual - current_coefficient * atom

        # 存储当前系数
        coefficients.append(current_coefficient)

    return coefficients

# 生成示例信号
np.random.seed(42)
signal_length = 1024
signal = np.random.randn(signal_length)

# 选择小波基（字典）
wavelet = 'db4'
# 将信号转换成二维数组
signal_2d = signal.reshape((1, signal_length))
print(signal_2d)
# 获取小波分解结果
coeffs = pywt.wavedec2(signal_2d, wavelet, mode='per')
for detail_coefficient in coeffs[1:]:
    print(detail_coefficient)
# 设置最大迭代次数的范围
max_iterations_range = range(1, 10)

# 评估不同 LEVEL 下的逼近误差
for max_iterations in max_iterations_range:
    # 运行 Matching Pursuit Algorithm
    coefficients = matching_pursuit_algorithm(signal, coeffs, max_iterations)

    # 从系数中重构信号
    reconstructed_signal = np.zeros_like(signal)
    for i in range(len(coefficients)):
        # 重新获取原子
        atom = pywt.waverec2([coeffs[0]], wavelet, mode='per')
        atom = atom.flatten()

        reconstructed_signal += coefficients[i] * atom

    # 计算逼近误差
    approximation_error = np.sum((signal - reconstructed_signal)**2)

    print(f"LEVEL: {max_iterations}, Approximation Error: {approximation_error}")




