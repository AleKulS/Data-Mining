import numpy as np
import math
import random as rand


def classifier(cov, mean1, mean2, mean3, P, x):
    ax1 = math.log(P) - 0.5 * np.dot((mean1*cov).T, mean1) + np.dot((x*cov).T, mean1)
    ax2 = math.log(P) - 0.5 * np.dot((mean2*cov).T, mean2) + np.dot((x*cov).T, mean2)
    ax3 = math.log(P) - 0.5 * np.dot((mean3*cov).T, mean3) + np.dot((x*cov).T, mean3)
    if ax1 > ax2 and ax1 > ax3:
        pred = 1
    elif ax2 > ax1 and ax2 > ax3:
        pred = 2
    else:
        pred = 3
    return(pred)


def bootstrap(cov, mean1, mean2, mean3, P, cl1, cl2, cl3):
    predboot = 0
    for x in range(len(cl1) * len(cl1)):
        if classifier(cov, mean1, mean2, mean3, P, cl1[rand.randint(0, len(cl1) - 1)]) == 1:
            predboot += 1
    for x in range(len(cl2) * len(cl2)):
        if classifier(cov, mean1, mean2, mean3, P, cl2[rand.randint(0, len(cl2) - 1)]) == 2:
            predboot += 1
    for x in range(len(cl3) * len(cl3)):
        if classifier(cov, mean1, mean2, mean3, P, cl3[rand.randint(0, len(cl3) - 1)]) == 3:
            predboot += 1
    return(predboot / ((len(cl1) * len(cl1))+(len(cl2) * len(cl2)) + (len(cl3) * len(cl3))))


P = 1/3     # Априорная вероятность всех классов (так как равновероятны)

cl1 = np.loadtxt('Sample1.txt')
cl2 = np.loadtxt('Sample2.txt')
cl3 = np.loadtxt('Sample3.txt')

main = np.copy(cl1)
main = np.append(main, cl2, axis = 0)  # Составляем общий массив данных
main = np.append(main, cl3, axis = 0)

mean1 = np.mean(cl1, axis = 0)
mean2 = np.mean(cl2, axis = 0)      # Высчитываем средние значения каждого класса
mean3 = np.mean(cl3, axis = 0)

cov = np.std(main) * np.std(main)    # Высчитываем "среднюю" матрицу ковариации

bootans = 0
for i in range(50):
    bootans += bootstrap(cov, mean1, mean2, mean3, P, cl1, cl2, cl3)    # Вызываем метод bootstrap
bootans = bootans / 50

predict = []
for x in main:
    predict.append(classifier(cov, mean1, mean2, mean3, P, x))    # Предсказываем классы

print (predict, '\n', 'Точность при проверкой bootstrap: ', bootans)

