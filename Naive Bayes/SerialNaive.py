# csv dosyalarını okumak için
import pandas as pd
import matplotlib.pyplot as plt

# csv dosyamızı okuduk.
data = pd.read_csv('C:\Repos\ParalellNaiveBayes\Databases\General\Iris.csv')

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
    
print(accuracy)
plt.plot(result)
plt.show()