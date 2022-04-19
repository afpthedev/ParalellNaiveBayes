# Yazar/author: Ahmet Faruk PALA

from msilib.schema import Class
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from multiprocessing import Pool
import os
from mpire import WorkerPool
#import ray
import matplotlib as plot
from sklearn.model_selection import train_test_split

# 3 tane farklı Naive Bayes Sınıfı vardır.
# GaussianNB : Tahmin edeceğiniz veri veya kolon sürekli (real,ondalıklı vs.) ise
# BernoulliNB : Tahmin edeceğiniz veri veya kolon ikili ise ( Evet/Hayır , Sigara içiyor/ İçmiyor vs.)
# MultinomialNB : Tahmin edeceğiniz veri veya kolon nominal ise ( Int sayılar )
# Duruma göre bu üç sınıftan birini seçebilirsiniz. Modelin başarı durumunu etkiler.
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB

#Confusion matris
from sklearn.metrics import accuracy_score, confusion_matrix
class serial:
    # MultinomialNB class in Serial
    def naive_bayes_multi(data):
        species = data.iloc[:,-1:].values
        x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,1:-1],species,test_size=0.33,random_state=0)
        multi = MultinomialNB()
        multi.fit(x_train, y_train.ravel())
        result = multi.predict(x_test)
        print('\n MultinomialNB Sonuclari\n')
        cm = confusion_matrix(y_test,result)
        print(cm)
        accuracy = accuracy_score(y_test, result)
        print(accuracy)
class paralell:

    # GaussianNB class in Naive Bayes
    def naive_bayes_gaus(data):
        species = data.iloc[:,-1:].values
        p= Pool()
        x_train, x_test, y_train, y_test = p.map(train_test_split(data.iloc[:,1:-1],species,test_size=0.33,random_state=0))
        gnb = GaussianNB()
        gnb.fit(x_train, y_train.ravel())
        result = gnb.predict(x_test)
        print(result)

    # GaussianNB class in Serial
    def naive_bayes_mp(data):
        start_time = time.time()
        p = Pool()
        result  = p.map(serial.naive_bayes_multi,data)
        p.close()
        p.join()
        end_time = time.time() - start_time
        print(f"Processing  took {end_time} time using multiprocessing of Gaussian Methods.")

    # BernoulliNB class in Serial
    def naive_bayes_ber(data):
        species = data.iloc[:,-1:].values
        x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,1:-1],species,test_size=0.33,random_state=0)
        ber = BernoulliNB()
        ber.fit(x_train, y_train.ravel())
        result = ber.predict(x_test)
        print('\n BernoulliNB Sonuclari\n')
        cm = confusion_matrix(y_test,result)
        print(cm)
        accuracy = accuracy_score(y_test, result)
        print(accuracy)





#Main Fonksiyon
if __name__ == '__main__':
    data = pd.read_csv('C:\Repos\ParalellNaiveBayes\Databases\General\Iris.csv')
    paralell.naive_bayes_mp(data)


