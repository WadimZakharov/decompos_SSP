import numpy as np
from sklearn import decomposition
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF
from ksvd import ApproximateKSVD

def creatMas(line):
    n=int(line[0])
    m=int(line[1])
    i,j = 0,0
    Mas=np.zeros((n,m))
    for ch in line[2:]:
        if ch != ' \n':
          Mas[i][j]=float(ch)
          j+=1
        else:
            j=0
            i+=1
    return Mas.T
def plot_ssp(cz, cz_reduced, zs, name):
    plt.figure()
    plt.plot(cz, zs)
    plt.plot(cz_reduced, zs)
    plt.legend(['Исходный вектор', 'Аппроксимация'])
    plt.ylabel('Глубина')
    plt.xlabel('Скорость')
    plt.title(name)
    plt.show()

def my_cross_val_score(CZ, n_comp = 3, model = 'pca', k = 5, random_state = False, shuffle = True ):
    if shuffle == True:
        save = np.copy(CZ)
        np.random.shuffle(CZ)
    kf = KFold(n_splits=k)
    cv = []
    pca = decomposition.PCA(n_components = n_comp, random_state = random_state)
    nmf = NMF(n_components = n_comp, random_state = random_state)
    kmeans = KMeans(n_clusters = n_comp, random_state = random_state)
    ksvd = ApproximateKSVD(n_components = n_comp)
    for train, test in kf.split(CZ):
        cz_test = CZ[test]
        cz_train = CZ[train]
        if model == 'pca':
            pca.fit(cz_train)
            CZ_reconstructed = pca.inverse_transform(pca.transform(cz_test))
        elif model == 'nmf':
            nmf.fit(cz_train)
            CZ_reconstructed = np.dot(nmf.transform(cz_test), nmf.components_)
        elif model == 'k_means':
            kmeans.fit(cz_train)
            CZ_reconstructed = kmeans.cluster_centers_[kmeans.predict(cz_test)]
        elif model == 'k_svd':
            meantr = np.mean(cz_train,axis=1)[:, np.newaxis]
            meantest = np.mean(cz_test, axis=1)[:, np.newaxis]
            dictionary = ksvd.fit(cz_train - meantr).components_
            gamma = ksvd.transform(cz_test - meantest)
            CZ_reconstructed = gamma.dot(dictionary) + meantest
        cv.append(mean_squared_error(CZ_reconstructed, cz_test)) 
    if shuffle == True:
        CZ = save
    return cv