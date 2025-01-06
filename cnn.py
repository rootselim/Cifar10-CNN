import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')

#load cifar10
(x_train,y_train),(x_test,y_test) = cifar10.load_data()
#veri seti normalizasyonu
x_train = x_train.astype('float32')/255
x_test = x_test.astype('float32')/255

#to_categorical
y_train = to_categorical(y_train,10)
y_test = to_categorical(y_test,10)

#Veri setini arttırma
datagen = ImageDataGenerator(
    rotation_range=20,#20 dereceye kadar dondurme saglar
    width_shift_range=0.2, #goruntuyu yatayda %20 kaydırır
    height_shift_range=0.2, #goruntuyu dikeyde %20 kaydırır
    shear_range=0.2, #goruntu uzerinde kaydırma
    zoom_range=0.2, #goruntuye zoom uygular
    horizontal_flip=True, #goruntuyu yatayda ters cevirme (simetrigini alma)
    fill_mode='nearest' #bos alanlari doldurmak icin en yakin pixel degerini kullan
)

datagen.fit(x_train)

model = Sequential()
model.add(Conv2D(32,(3,3),activation='relu',padding="same",input_shape=(32,32,3)))
#burda evirisimli sinir aginin giris boyutunun (32,32,3) olmak uzere 3 boyuttan olustugunu belirttik
#32,(3,3) feauture extraction icin gerekli olan parametredir 32 adet (3,3) luk filtreler olusturduk burda amac verinin onemli parcalarini elde edebilmek

model.add(Conv2D(32,(3,3),activation='relu')) #goruntu uzerindeki yerel yani onemli ozellikleri çıkarmak icin 32 adet (3,3) boyutunda filtre kullanıldı
model.add(MaxPooling2D(pool_size=(2,2))) #burda ise evrisim katmanının çıktılarını daha kucuk boyutlara indirerek model maliyetini dusurur
model.add(Dropout(0.25)) #baglantıların %25ini rastgele kapat

model.add(Conv2D(64,(3,3),activation='relu',padding="same"))
model.add(Conv2D(64,(3,3),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten()) #burda flatten kullanılmasınıın amacı fully connected katmanlar kullanılmadan once verinin tek boyuta indirilmesi gerekmektedir
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10,activation='softmax'))

model.compile(optimizer = RMSprop(learning_rate=0.0001,decay=1e-6),loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(datagen.flow(x_train,y_train,batch_size=64),epochs=50,validation_data=(x_test,y_test))












































