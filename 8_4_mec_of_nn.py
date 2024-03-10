import numpy as np
import math 

# binary cls
def memorize_bin_cls(data: np.ndarray, labels: np.ndarray):
    thresholds = 0
    N, D = data.shape
    table = np.zeros((N, 2))
    for row in range(N):
        table[row, :] = [data[row].sum(), labels[row]]
    sorted_table = table[table[:, 0].argsort()]
    curr_class = 0

    for row in range(N):
        if not sorted_table[row, 1] == curr_class:
            curr_class = sorted_table[row, 1]
            thresholds += 1

    min_threshold = np.log2(thresholds + 1)
    mec = min_threshold * (D + 1) + (min_threshold + 1)
    return mec

# multi-class cls
def memorize_multi_cls(data: np.ndarray, labels: np.ndarray):
    thresholds = 0
    N, D = data.shape
    table = np.zeros((N, 2))
    n_class = len(np.unique(labels))
    for row in range(N):
        table[row, :] = [data[row].sum(), labels[row]]
    sorted_table = table[table[:, 0].argsort()]
    curr_class = 0

    for row in range(N):
        if not sorted_table[row, 1] == curr_class:
            curr_class = sorted_table[row, 1]
            thresholds += 1

    thresholds *= math.ceil(math.log2(n_class))
    min_threshold = np.log2(thresholds + 1) 
    mec = min_threshold * (D + 1) + (min_threshold + 1)
    return mec

# Higuchi FD
def hFD(a, k_max=2**3): 
    L = []
    x = []
    N = len(a)
    for k in range(1,k_max):
        Lk = 0
        for m in range(0,k):
            #we pregenerate all idxs
            idxs = np.arange(1,int(np.floor((N-m)/k)),dtype=np.int32)
            Lmk = np.sum(np.abs(a[m+idxs*k] - a[m+k*(idxs-1)]))
            Lmk = (Lmk*(N - 1)/(((N - m)/ k)* k)) / k
            Lk += Lmk
        L.append(np.log(Lk/(m+1)))
        x.append([np.log(1.0/ k), 1])
    (p, r1, r2, s)=np.linalg.lstsq(x, L)
    return p[0]

# regression
def memorize_reg(data: np.ndarray, labels: np.ndarray):
    thresholds = 0
    N, D = data.shape
    table = np.zeros((N, 2))

    # calculate fractal dimension
    fd = hFD(labels)
    labels = np.floor(np.emath.logn(fd, labels))
    n_class = len(np.unique(labels))
    
    for row in range(N):
        table[row, :] = [data[row].sum(), labels[row]]
    sorted_table = table[table[:, 0].argsort()]
    curr_class = 0

    for row in range(N):
        if not sorted_table[row, 1] == curr_class:
            curr_class = sorted_table[row, 1]
            thresholds += 1

    thresholds *= math.ceil(math.log2(n_class))
    min_threshold = np.log2(thresholds + 1) 
    mec = min_threshold * (D + 1) + (min_threshold + 1)
    return mec

if __name__ == "__main__":
    data = np.random.random((5, 4))
    labels = np.random.random((5,))
    memorize_bin_cls(data, labels)
    memorize_multi_cls(data, labels)
    memorize_reg(data, labels)
