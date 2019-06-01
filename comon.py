import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
from keras.models import Model
from keras.layers import Input, Dense, Dropout
from sklearn.cluster import KMeans
from ksvd import ApproximateKSVD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

def AEncoder(input_dim, latent_dim):
    hidden_layer = int(input_dim * 0.8)
    
    inputs = Input(shape=(input_dim,))
    inp_layar1 = Dense(hidden_layer, activation='softplus',kernel_initializer='he_normal')(inputs)
    #np_layar1 = Dropout(0.1)(inp_layar1)
    
    encoded = Dense(latent_dim, activation='softplus', kernel_initializer='he_normal')(inp_layar1)
    
    dec_lay = Dense(hidden_layer,activation='linear', kernel_initializer='he_normal')(encoded)
    #dec_lay = Dropout(0.1)(dec_lay)
    decoded = Dense(input_dim,activation='linear', kernel_initializer='he_normal')(dec_lay)
    
    autoencoder = Model(inputs, decoded)
    encoder = Model(inputs, encoded)
    
    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder, encoder

def researcher_pca(CLF,X_train, Y_train, X_test, Y_test, n_comp=[3,5,7,10], **kwargs):
    scores = []
    for n in n_comp:
        dec = PCA(n_components=n)
        clf = CLF(**kwargs)
        pipe = Pipeline([('dec', dec), ('clf', clf)])
        pipe.fit(X_train, Y_train)
        predict = pipe.predict(X_test)
        score = accuracy_score(Y_test, predict)
        scores.append(score)
    return scores


def researcher_kmeans(CLF,X_train, Y_train, X_test, Y_test, n_comp=[3,5,7,10], **kwargs):
    scores = []
    for n in n_comp:
        kmeans = KMeans(n_clusters = n)
        kmeans.fit(X_train)
        
        test_red = np.zeros((len(X_test), n))
        train_red = np.zeros((len(X_train), n))

        centers_test=kmeans.predict(X_test)
        centers_train=kmeans.predict(X_train)
        
        for i in range(len(X_train)):
            train_red[i,centers_train[i]]=1
        
        for i in range(len(X_test)):
            test_red[i,centers_test[i]]=1
        
        clf = CLF(**kwargs)
        clf.fit(train_red, Y_train)
        predict = clf.predict(test_red)
        score = accuracy_score(Y_test, predict)
        scores.append(score)
    return scores



def researcher_ksvd(CLF, X_train, Y_train, X_test, Y_test, k=2, n_comp=[3,5,7,10], **kwargs):
    scores = []
    for n in n_comp:
        ksvd = ApproximateKSVD(n_components=n, transform_n_nonzero_coefs=max(n-k, 1))
        meantr = np.mean(X_train,axis=0)
        ksvd.fit(X_train - meantr).components_
        gamma_train = ksvd.transform(X_train - meantr)
        gamma_test = ksvd.transform(X_test - meantr)
        
        clf = CLF(**kwargs)
        clf.fit(gamma_train, Y_train)
        predict = clf.predict(gamma_test)
        score = accuracy_score(Y_test, predict)
        scores.append(score)
    return scores


def researcher_ae(CLF, X_train, Y_train, X_test, Y_test, n_units=[3,5,7,10],epochs=750, **kwargs):
    scores = []
    scaler = StandardScaler()
    X_train_std = scaler.fit_transform(X_train)
    X_test_std = scaler.transform(X_test)
    N_cord = X_train.shape[1]
    for n in n_units:
        autoencoder_std, encoder_std = AEncoder(N_cord, n)
        autoencoder_std.fit(X_train_std, X_train_std,
                epochs=epochs,
                batch_size=64,
                shuffle=True, verbose=0)
          
        test_embedding = encoder_std.predict(X_test_std)
        train_embedding = encoder_std.predict(X_train_std)
        
        clf = CLF(**kwargs)
        clf.fit(train_embedding, Y_train)
        predict = clf.predict(test_embedding)
        score = accuracy_score(Y_test, predict)
        scores.append(score)
    return scores

def plot_confusion_matrix(cm, classes=['winter', 'summer'],
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()

def print_metrics(clf, X_train, X_test, y_train, y_test):
    predict_train = clf.predict(X_train)
    predict_test = clf.predict(X_test)
    print('Accuracy на тренировочных данных: {}'.format(accuracy_score(y_train, predict_train)))
    print('Accuracy на тестовых данных: {}'.format(accuracy_score(y_test, predict_test)))
    
    plot_confusion_matrix(confusion_matrix(y_train, predict_train), title='Confusion matrix для обучающих данных')
    plot_confusion_matrix(confusion_matrix(y_test, predict_test), title='Confusion matrix для тестовых данных')
    
def feature_importances(clf, zs, size = (10,12), title_x = 'Важность координаты', title_y = 'Глубина'):
    fi = clf.feature_importances_
    x = range(1,len(fi)+1)
    plt.figure(figsize=size)
    plt.barh(x, fi)
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    plt.yticks(x,list(map(str,zs)))
    plt.show()



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
    print("MSE:",mean_squared_error(cz, cz_reduced))
    plt.figure()
    plt.plot(cz, zs)
    plt.plot(cz_reduced, zs)
    plt.legend(['Исходный вектор', 'Аппроксимация'])
    plt.ylabel('Глубина')
    plt.xlabel('Скорость')
    plt.title(name)
    plt.show()
    

    
    
def creat_data_month(CZ, N='all'):
    X, Y = [], []
    i = 0
    for cz in CZ:
        if N=='all':
            N=len(cz)
        X += list(cz)[:N]
        Y += [i]*N
        i += 1
    return np.array(X), np.array(Y)

def creat_win_sum_data(CZ, winter_index, summer_index, N='all'):
    X, Y = [], []
    for ind in winter_index:
        if N=='all':
            N=len(CZ[ind-1])
        X += list(CZ[ind-1])[:N]
    n = len(X)
    Y += [0 for i in range(n)]
    
    for ind in summer_index:
        if N=='all':
            N=len(CZ[ind-1])
        X += list(CZ[ind-1])[:N]
    Y += [1 for i in range(len(X)-n)]
    return np.array(X), Y

def plot_confusion_matrix(cm, classes=['winter', 'summer'],
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()