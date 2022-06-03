import random
import time
from multiprocessing import Process,Pool
import os
from mpire import WorkerPool#
from joblib import Parallel, delayed
import ray 
# csv dosyalarını okumak için
import pandas as pd
import matplotlib.pyplot as plt


def pool_naive(data):
    number = range(151)
    start_time = time.time()
    p = Pool()
    result = p.map(naive_bayes, data)
    p.close()
    p.join()
    end_time = time.time() - start_time
    print(f"Processing {len(data)} numbers took {end_time} time using Pool Library multiprocessing.")

def naive_bayes(data):
    start_time = time.time()
    # Bağımlı Değişkeni (species) bir değişkene atadık
    # 4 Finans için  -1 ise Irıs veritabanı için
    species = data.iloc[:,-1:].values
    # Veri kümemizi test ve train şeklinde bölüyoruz
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,1:-1],species,test_size=0.50,random_state=0)   
    from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
     # GaussianNB sınıfından bir nesne ürettik
    gnb = GaussianNB()   
    # Makineyi eğitiyoruz. 
    #Ravel diziyi düzlemselleştirir.
    gnb.fit(x_train, y_train.ravel())   
    # Test veri kümemizi verdik ve iris türü tahmin etmesini sağladık
    result = gnb.predict(x_test)   
    # Karmaşıklık matrisi
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test,result)
    print(" Konfusyon matrisi\n")
    print(cm)
    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, result) 
    end_time = time.time() - start_time
    print(f"Processing {len(data)} numbers took {end_time} time using serial processing.")
    
# Multiprocessing using Process LİB
def process_naive(data):
    number = range(151)
    start_time = time.time()
    p = Process()
    print(Process(target=naive_bayes,args=(data)))
    p.start()
    p.join()
    end_time = time.time() - start_time
    print(f"Processing {len(data)} numbers took {end_time} time using multiprocessing.")


def naive_bayes_with_dask(data):
    from dask_ml import datasets
    from dask_mk.naive_bayes import GaussianNB as gnb1
    X, y = data.make_classification(chunks=50)
    gnb = gnb1()
    gnb.fit(X,y)
    
    
    

if __name__ == '__main__':
    # csv dosyamızı okuduk.
    data = pd.read_csv('C:\Repos\ParalellNaiveBayes\Databases\General\Iris.csv')
    #process_naive(data)
    #naive_bayes(data)
    #pool_naive(data)
    naive_bayes_with_dask(data)