
'''
İmport edilmesi gerekenleri ekliyoruz.
'''
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
import NaiveNative

class NaiveBayes:
    def naive_bayes():
        data = pd.read_csv('C:\Repos\ParalellNaiveBayes\Databases\General\Breast_cancer_data.csv')          
        start_time = time.time()
        print('Classic Naive Bayes')
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
    
    def NaiveBayesByAFP(): # AFP tarafınan yazılan naive bayes kodudur.
        data = pd.read_csv('C:\Repos\ParalellNaiveBayes\Databases\General\Breast_cancer_data.csv')  
        print('Native Naive by AFP ')           
        starttime = time.time()
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(data, test_size=.4, random_state=41)
            
        X_test = test.iloc[:,:-1].values
        Y_test = test.iloc[:,-1].values
        Y_pred = NaiveNative.naive_bayes_categorical(train, X=X_test, Y="diagnosis")        
        from sklearn.metrics import confusion_matrix, f1_score
        print(confusion_matrix(Y_test, Y_pred))
        print('Confusion matris Result')
        print(1- f1_score(Y_test, Y_pred))
        end_time  = time.time() - starttime
        print(f"Processing {len(data)} numbers took {end_time} time using Native --- AFP ---- Bayes serial processing.")
        print( )
        return end_time
class DıstNaiveBayes:      
    def naive_bayes_with_dask():
        print('Naive Bayes by Dask D.S')
        start_time = time.time()
        from dask_ml import datasets
        from dask_ml.naive_bayes import GaussianNB as gnb1
        X, y = datasets.make_classification(chunks=50)
        gnb1 = gnb1()
        gnb1.fit(X,y)
        end_time = time.time() - start_time
        print(f"Processing {len(data)} numbers took {end_time} time using Dask multiprocessing.")
        return end_time
        
    def process_naive():#By sklearn
        data = pd.read_csv('C:\Repos\ParalellNaiveBayes\Databases\General\Breast_cancer_data.csv')  
        print('Naive BAyes by Process D.S')
        number = range(151)
        start_time = time.time()
        p = Process()
        print(Process(target=NaiveBayes.naive_bayes,args=(data)))
        p.start()
        p.join()
        end_time = time.time() - start_time
        print(f"Processing {len(data)} numbers took {end_time} time using Process multiprocessing.")
        return end_time
    
    def naive_bayes_with_daskAFP():# AFP 
        print('Naive Bayes by Dask D.S')
        start_time = time.time()
        from dask_ml import datasets
        from dask_ml.naive_bayes import GaussianNB as gnb1
        X, y = datasets.make_classification(chunks=50)
        gnb1 = gnb1()
        gnb1.fit(X,y)
        end_time = time.time() - start_time
        print(f"Processing {len(data)} numbers took {end_time} time using Dask multiprocessing.--- AFP----")
        return end_time
        
    def process_naiveAFP():#By AFP
        data = pd.read_csv('C:\Repos\ParalellNaiveBayes\Databases\General\Breast_cancer_data.csv')  
        print('Naive BAyes by Process D.S AFP ')
        number = range(151)
        start_time = time.time()
        p = Process()
        print(Process(target=NaiveBayes.NaiveBayesByAFP(),args=(data)))
        p.start()
        p.join()
        end_time = time.time() - start_time
        print(f"Processing {len(data)} numbers took {end_time} time using Process multiprocessing..--- AFP----")
        return end_time
    
if __name__ == "__main__":        
    data = pd.read_csv('C:\Repos\ParalellNaiveBayes\Databases\General\Breast_cancer_data.csv')  
  
    #Serial processing By Naive Bayes AFP 
    NaiveBayes.NaiveBayesByAFP()
    
    #Serial processing By Naive Bayes Sklearn 
    NaiveBayes.naive_bayes()
    
    
    #Parallel processing By Naive Bayes AFP 
    DıstNaiveBayes.naive_bayes_with_daskAFP()
    DıstNaiveBayes.process_naiveAFP()
    
    #Parallel processing By Naive Bayes Sklearn 
    DıstNaiveBayes.naive_bayes_with_dask()
    DıstNaiveBayes.process_naive()
    
   
    '''
    #Plot işlemleri için 
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    names = ['Sklearn Naive Bayes',' AFP Naive Bayes','Paralel Sklearn Naive Bayes ','Paralel Sklearn AFP Naive Bayes']
    value1 = [NaiveBayes.naive_bayes(),NaiveBayes.NaiveBayesByAFP(),DıstNaiveBayes.process_naive(),DıstNaiveBayes.process_naiveAFP()]
    ax.bar(names,value1)
    plt.title('PAralel veya Seri haldeki Naive Bayes kodlarının Process Kütüphanesi süreleri')
    plt.show()
    
    '''
    #Plot 
    import matplotlib.pyplot as plt
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    plt.plot(label='First Line')
    plt.plot(label='Second Line')
    plt.title('Breast Cancer datasetinde Naive Bayes paralel ve  kodlarının süre durumu')
    langs = ['Serial Naive Bayes', 'Serial Naive Bayes-AFP-','Parallel NaiveBayes', 'Parallel NaiveBayes-AFP-']
    students = [NaiveBayes.naive_bayes(),NaiveBayes.NaiveBayesByAFP(),DıstNaiveBayes.process_naive(),DıstNaiveBayes.process_naiveAFP()]
    ax.bar(langs,students)
    
    
    plt.show()
