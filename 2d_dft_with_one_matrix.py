import numpy as np

def dft_matrix_2d(N, M):
    """
    Generate a 2D DFT matrix of size N x M
    """
    n = np.arange(N).reshape((N, 1))
    m = np.arange(M)
    u, v = np.meshgrid(m, n)
    dft_mat = np.exp(-2j * np.pi * (u * v / M)) / np.sqrt(N * M)
    return dft_mat

def IDFT_matrix_2d(N, M):
    """
    Generate a 2D IDFT matrix of size N x M
    """
    n = np.arange(N).reshape((N, 1))
    m = np.arange(M)
    u, v = np.meshgrid(m, n)
    idft_mat = np.exp(2j * np.pi * (u * v / M))
    return idft_mat

def DFT2D(image):
    """
    Perform 2D DFT using a DFT matrix
    """
    N, M = image.shape
    DFT_mat = dft_matrix_2d(N, M)
    dft_image = np.dot(DFT_mat, image)
    return dft_image

def IDFT2D(X):
    """
    Perform 2D IDFT using a IDFT matrix
    """
    N, M = X.shape
    IDFT_mat = IDFT_matrix_2d(N, M)
    idft_image = np.dot(IDFT_mat, X)
    return idft_image

def main():
    image = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 90, 12],
                      [13, 14, 15, 900]])


    print("Original Image:")
    print(image)

    dft_result = DFT2D(image)
    print("\n2D DFT Result (Complex Numbers):")
    print(dft_result)

    idft_result = IDFT2D(dft_result)
    print("\n2D IDFT Result (Complex Numbers):")
    print(idft_result)

    print("\nReconstructed Image:")
    print(idft_result.real)

if __name__ == "__main__":
    main()