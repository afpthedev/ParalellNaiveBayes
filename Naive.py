# csv dosyalarını okumak için
import pandas as pd

# csv dosyamızı okuduk.
data = pd.read_csv('Iris.csv')

# Bağımlı Değişkeni ( species) bir değişkene atadık
species = data.iloc[:,-1:].values

# Veri kümemizi test ve train şekinde bölüyoruz
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data.iloc[:,1:-1],species,test_size=0.33,random_state=0)


# GaussianNB sınıfını import ettik
# 3 tane farklı Naive Bayes Sınıfı vardır.
# GaussianNB : Tahmin edeceğiniz veri veya kolon sürekli (real,ondalıklı vs.) ise
# BernoulliNB : Tahmin edeceğiniz veri veya kolon ikili ise ( Evet/Hayır , Sigara içiyor/ İçmiyor vs.)
# MultinomialNB : Tahmin edeceğiniz veri veya kolon nominal ise ( Int sayılar )
# Duruma göre bu üç sınıftan birini seçebilirsiniz. Modelin başarı durumunu etkiler.
from sklearn.naive_bayes import GaussianNB

# GaussianNB sınıfından bir nesne ürettik
gnb = GaussianNB()

# Makineyi eğitiyoruz
gnb.fit(x_train, y_train.ravel())

# Test veri kümemizi verdik ve iris türü tahmin etmesini sağladık
result = gnb.predict(x_test)

# Karmaşıklık matrisi
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,result)
print(cm)

# Başarı Oranı
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, result)
# Sonuç : 0.96
print(accuracy)
