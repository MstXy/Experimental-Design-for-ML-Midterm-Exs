from sklearn.neighbors import KNeighborsClassifier
import numpy as np

N = 1000
D = 100
l,r = -10, 10

def info_cap(n_cls=2):
    X = np.random.uniform(l, r, size=(N,1))
    for _ in range(1, D):
        x = np.random.uniform(l, r, size=(N,1))
        X = np.concatenate([X, x], axis=1)
    y = np.random.choice(range(n_cls), size=(N,))

    n_elim = 0
    for i in range(N):
        nn = KNeighborsClassifier(n_neighbors=1)
        nn.fit(np.concatenate((X[:i, :],X[i+1:, :]), axis=0), np.concatenate((y[:i],y[i+1:]), axis=0))
        if nn.predict(X[i:i+1, :]) == y[i]:
            n_elim += 1
    
    if n_cls == 2:
        print("For binary classification, \nInformation capacity of NN with data dimension {}, total points {} is {:.2f}".format(D, N, N/(N-n_elim)))
    else:
        print("For {}-class classification, \nInformation capacity of NN with data dimension {}, total points {} is {:.2f}".format(n_cls, D, N, N/(N-n_elim)))

info_cap(2)
info_cap(5)
info_cap(10)
