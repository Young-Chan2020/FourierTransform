import numpy as np

#accelerate computation via matrix multiple
def DFT_kernel(x):
    # sampling point
    N = len(x)
    #[0,N-1]
    n = np.arange(N)
    #[0,N-1]^T frequency sampling point
    k = n.reshape((N, 1))
    #base function matrix size=N*N    row denotes the freqency;column denotes the samplying point
    e = np.exp(-2j * np.pi * k * n / N)
    return e,e.T
def iDFT_kernel(x):
    # sampling point
    N = len(x)
    #[0,N-1]
    n = np.arange(N)
    #[0,N-1]^T frequency sampling point
    k = n.reshape((N, 1))
    #base function matrix size=N*N    row denotes the freqency;column denotes the samplying point
    e = np.exp(2j * np.pi * k * n / N)
    return e,e.T
if __name__ == '__main__':
    image = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 90, 12],
                      [13, 14, 15, 900]])
    image=np.random.random([4,4])
    dft_kernel,dft_kernel_T = DFT_kernel(image)
    iDFT_kernel,iDFT_kernel_T = iDFT_kernel(image)
    print(image)

    dft_matrix = np.dot(np.dot(dft_kernel, image), dft_kernel_T)

    reconstruct = np.dot(np.dot(iDFT_kernel_T, dft_matrix), iDFT_kernel_T).real/(image.shape[0]*image.shape[1])

    print(reconstruct)
    print(np.allclose(reconstruct, image))




    dft_kernel_inverse=np.linalg.inv(dft_kernel)
    dft_kernel_conj=np.conj(dft_kernel)
    print("")
    print(dft_kernel_conj)
    print("")
    print(iDFT_kernel)
    print(dft_kernel_conj==iDFT_kernel)