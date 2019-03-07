import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix
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

def creat_win_sum_data(CZ, winter_index, summer_index):
    X, Y = [], []
    for ind in winter_index:
        X += list(CZ[ind-1])
    n = len(X)
    Y += [0 for i in range(n)]
    
    for ind in summer_index:
        X += list(CZ[ind-1])
    Y += [1 for i in range(len(X)-n)]
    return X, Y

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
    
def feature_importances(clf, size = (10,12), title_x = 'Важность координаты', title_y = 'Номер координаты'):
    fi = clf.feature_importances_
    x = range(1,len(fi)+1)
    plt.figure(figsize=size)
    plt.barh(x, fi)
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    plt.show()