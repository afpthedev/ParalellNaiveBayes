import random
import time
from multiprocessing import Process,Pool
import os
from mpire import WorkerPool
from joblib import Parallel, delayed
#import ray 
# csv dosyalarını okumak için
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from dask_ml import datasets
######IRASSSS
def pool_naive(data):
    number = range(151)
    start_time = time.time()
    p = Pool()
    result = p.map(naive_bayes, data)
    p.close()
    p.join()
    end_time = time.time() - start_time
    print(f"Processing {len(data)} numbers took {end_time} time using Pool Library multiprocessing.")

def naive_bayes(datasets):
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
    return end_time
   # Multiprocessing using Process LİB
def process_naive():
    data = pd.read_csv('C:\Repos\ParalellNaiveBayes\Databases\General\Iris.csv')
    number = range(151)
    start_time = time.time()
    p = Process()
    print(Process(target=naive_bayes,args=(data)))
    p.start()
    p.join()
    end_time = time.time() - start_time
    print(f"Processing {len(data)} numbers took {end_time} time using Process multiprocessing.")
    return end_time

def naive_bayes_with_dask():
    start_time = time.time()
    from dask_ml import datasets
    from dask_ml.naive_bayes import GaussianNB as gnb1
    X, y = datasets.make_classification(chunks=50)
    gnb1 = gnb1()
    gnb1.fit(X,y)
    end_time = time.time() - start_time
    print(f"Processing {len(data)} numbers took {end_time} time using Dask multiprocessing.")
    return end_time
    
    

if __name__ == "__main__":
    #csv dosyamızı okuduk.
    data = pd.read_csv('C:\Repos\ParalellNaiveBayes\Databases\General\Iris.csv')
    data1 = pd.read_csv('C:\Repos\ParalellNaiveBayes\Databases\General\Breast_cancer_data.csv')
    process_naive()
    naive_bayes(data)
    #pool_naive(data)#çalışmıyor bu 
    
    naive_bayes_with_dask()
    
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.title('Paralel haldeki Naive Bayes kodlarının kütüphane performansları')
    ax = fig.add_axes([0,0,1,1])
    langs = ['Dask', 'Process']
    plt.title('Paralel haldeki  Naive Bayes kodları için kullanılan kütüphanelerin sürelerin karşılaştırılaması')
    students = [naive_bayes_with_dask(),process_naive()]
    ax.bar(langs,students)
    plt.show()
    

