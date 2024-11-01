import numpy as np
def DFT(x):
    """
    Function to calculate the
    discrete Fourier Transform
    of a 1D real-valued signal x
    """
    # sampling point
    N = len(x)
    #[0,N-1]
    n = np.arange(N)
    #[0,N-1]^T frequency sampling point
    k = n.reshape((N, 1))
    #base function matrix size=N*N    row denotes the freqency;column denotes the samplying point
    e = np.exp(-2j * np.pi * k * n / N)
    #X[k] is the amplitude of which base is of frequency k
    X = np.dot(e, x)

    return X


def IDFT(X):
    """
    Function to calculate the inverse discrete Fourier Transform
    of a 1D complex-valued signal X
    """
    # sampling point
    N = len(X)
    # [0,N-1]
    n = np.arange(N)
    # [0,N-1]^T frequency sampling point
    k = n.reshape((N, 1))

    e = np.exp(2j * np.pi * k * n / N)  # 计算复指数矩阵

    x = np.dot(e, X)  # 计算矩阵乘法，得到时域信号
    x = x / N  # 归一化因子1/N

    return x
if __name__ == '__main__':
    x = np.array([1, 90, 1])
    print(f"origin signal is {x}")
    X=DFT(x)
    print(f"coef matrix is {X}")
    x_reconstructed = np.real(IDFT(X))
    print(f"reconstucted signal is {x_reconstructed}")  # 应该输出接近原始信号的值
