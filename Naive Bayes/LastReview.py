# -*- coding: utf-8 -*-
"""
Created on Sat May 21 15:46:23 2022

@author: yediz
"""


import NaiveNative
import time 
import pandas as pd 
import matplotlib.pyplot as plt

def NaiveBayesByAFP(): # AFP tarafınan yazılan naive bayes kodudur.
    data = pd.read_csv('C:\Repos\ParalellNaiveBayes\Databases\General\Breast_cancer_data.csv')
    print('Native Naive')           
    starttime = time.time()
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(data, test_size=.2, random_state=41)
        
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

def Serial_Native_By_Sklearn():# Sklearn kütüphanesi kullanılarak oluşturulan fonksiyondur  

        starttime = time.time()
        # csv dosyamızı okuduk.
        data = pd.read_csv('C:\Repos\ParalellNaiveBayes\Databases\General\Breast_cancer_data.csv')
        # Bağımlı Değişkeni (species) bir değişkene atadık
        # 4 Finans için  -1 ise Irıs veritabanı için
        species = data.iloc[:,-1:].values
        # Veri kümemizi test ve train şeklinde bölüyoruz
        from sklearn.model_selection import train_test_split
        x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,1:-1],species,test_size=0.33,random_state=0)

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
        print(cm)


        from sklearn.metrics import accuracy_score
        accuracy = accuracy_score(y_test, result)

        # Sonuç: 0.96 GaussianNB
        # Sonuç: 0.3 BernoulliNB
        # Sonuç: 0.7  MultinomalNB
        end_time  = time.time() - starttime
        print(f"Processing {len(data)} numbers took {end_time} time using Serial  Bayes serial processing.")
        print( )
        print(accuracy)
      
        return end_time


# Data
data1 = pd.read_csv('C:\Repos\ParalellNaiveBayes\Databases\General\Breast_cancer_data.csv')

#Plot 
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
plt.plot(label='First Line')
plt.plot(label='Second Line')

langs = ['Serial Naive Bayes', 'Native Naive Bayes',]
students = [Serial_Native_By_Sklearn(),NaiveBayesByAFP()]
ax.bar(langs,students)
plt.title('Seri haldeki Naive Bayes kodlarının süre durumu')
plt.show()
