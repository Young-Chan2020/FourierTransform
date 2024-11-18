import numpy as np

def dft_matrix(N, M):
    n, k = np.meshgrid(np.arange(N), np.arange(M))
    W = np.exp(-2j * np.pi * n * k / N)
    return W

def idft_matrix(N, M):
    n, k = np.meshgrid(np.arange(N), np.arange(M))
    W = np.exp(2j * np.pi * n * k / N)
    return W

def dft(x, M):
    N = len(x)
    W = dft_matrix(N, M)
    return np.dot(W, x)

def idft(X, M, origin_samplingPoint):
    W = idft_matrix(origin_samplingPoint, M)
    return np.dot(X.T, W) / (M)

# 测试代码
x = np.array([0, 1, 2, 3,0, 1, 2, 3,0, 1, 2, 3,0, 1, 2, 3])
origin_samplingPoint = len(x)
M = 4000  # 基本函数的数量
print("信号：", x)
# 计算DFT
X = dft(x, M)
print("DFT结果：", X)

# 计算IDFT
x_reconstructed = idft(X, M, origin_samplingPoint)
np.set_printoptions(suppress=True, precision=5)
print("逆DFT结果：", x_reconstructed.real)