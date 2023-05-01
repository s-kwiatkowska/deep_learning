#!/usr/bin/env python
# coding: utf-8

# # Dobre praktyki z DeepLearning 
# 
# ### Celem jest wskazanie kilku dobrych praktyk w DL (w szczególności z `Keras` i/lub `Tensorflow`). 

# ## Krok po kroku
# 
# Jeśli dopiero zaczynasz przygodę z Python, to może potrzebujesz dodatkowego wsparcia i wyjaśnienia składni lub tego, co robi ta czy inna funkcja. Jeśli tak, to poniższe wideo jest dla Ciebie. Krok po kroku przechodzę po notebooku i wyjaśniam, co tu się dzieje.

# In[4]:


get_ipython().run_cell_magic('html', '', '<iframe style="height:500px;width:100%" src="https://bit.ly/3bvjqII" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>')


# ## Callbacks
# 
# Keras posiada sporo różnych "callbacków", również można zdefiniować swoje własne. Natomiast warto wiedzieć co najmniej o tych:
# - [EarlyStopping](https://bit.ly/2RW4BrR) - umożliwia zatrzymanie trenowania modeli jeśli po X epokach model tylko pogarsza się (następuje overfitting).
# - [ModelCheckpoint](https://bit.ly/3omctim) - umożliwia zapisywanie modelu, szczególnie zależy nam na zapisywaniu najlepszej wersji (dla którejś epoki, wtedy nie musimy uczyć modelu od nowa).
# - [ReduceLROnPlateau](https://bit.ly/3tW21Pz) - umożliwia zmniejszenie learning rate, kiedy model zatrzymał się (to czasem może pomóc).
# - [PlotLossesCallback](https://bit.ly/3eMQ2j7) - rysowanie krzywej uczenia się w łatwiejszy sposób.
# - [TensorBoard](https://bit.ly/2RlNBLz) - umożliwia podpięcie się do `TensorBoard` (pod warunkiem, że backend mamy `TensorFlow`).

# In[40]:


import os

from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard

import catboost as ctb
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

import numpy as np


# In[41]:


(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape, X_test.shape


# In[42]:


# najpierw spłaszczamy
if len(X_train.shape) == 3:
    num_pixels = X_train.shape[1] * X_train.shape[2]
    X_train = X_train.reshape(X_train.shape[0], num_pixels).astype("float32")
    X_test = X_test.reshape(X_test.shape[0], num_pixels).astype("float32")

print(X_train.shape, X_test.shape)

# skalujemy od 0 do 1
if np.max(X_train) > 1: X_train /= 255
if np.max(X_test) > 1: X_test /= 255

# one-hot encoding dla zmiennej docelowej
if len(y_test.shape) == 1:
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    num_classes = y_test.shape[1]


# In[44]:


model = Sequential([
    Dense(512, input_dim=num_pixels, activation='relu'),
    Dense(num_classes, kernel_initializer='normal', activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# Tworzymy katalog, gdzie będą zapisywane informacje podczas trenowania modelu (będzie potrzebne np. dla `tensorboard`).

# In[45]:


LOG_DIR = "../logs/tensorboard"
get_ipython().system('mkdir -p "$LOG_DIR"')


# Ustawiamy kilka różnych callback'ów.

# In[46]:


callbacks = [ 
    EarlyStopping(patience=4), #jeśli 4 epoki z rzędu nie ma poprawy, to zatrzymaj się
    ModelCheckpoint('../output/model.best.hdf5', save_best_only=True), # zapis modelu po każdej epoce
    ReduceLROnPlateau(patience=3), #jeśli 3 epoki z rzędu nie ma poprawy, zmniejsz krok (learning_rate)
    TensorBoard(log_dir=LOG_DIR), # odkładanie logów, aby można było użyć Tensorboard
]

history = model.fit(X_train, y_train,
          batch_size=1024, epochs=5, verbose=1,
          validation_data=(X_test, y_test),
          callbacks=callbacks)


# Model zatrzymał się przed wykonaniem 100 epok. Widać, że przed zakończeniem zaczęło następować wypłaszczanie wyników modelu. Najlepszy model zapisał się w folderze `output`. 
# 
# **Zwróć uwagę**, że w tej chwili zmienna `model` posiada stan ostatniej epoki, która niekoniecznie posiada najlepszy model (prawie nigdy). Dlatego to, co zrobimy teraz, to wczytamy zapisane wagi z `../output/model.best.hdf5`.
# 
# Możemy wczytać cały model od zera za pomocą `load_model` albo tylko same wagi za pomocą `.load_weights`.

# In[47]:


model = load_model('../output/model.best.hdf5')


# W tej chwili mamy załadowany najlepszy model i możemy robić predykcję.

# ## TensorBoard
# 
# Możemy sprawdzić krzywą uczenia się również przy pomocy `TensorBoard`.  
# ???
# 
# Uruchomiamy TB np. na porcie 8050 (mało istotny jest numerek).
# 
# Najpierw uruchom komorke poniżej.

# In[62]:


get_ipython().system('tensorboard --logdir=../logs/tensorboard --port=8050 --host=0.0.0.0')


# #  🚨🚨🚨  Następnie kliknij 👉👉👉 [tutaj](/hub/user-redirect/proxy/8050/) 👈👈👈

# Zauważ, że poprzednia komórka będzie działać w nieskończoność, chyba że ją zatrzymamy.
# 
# Dlatego zanim przejdziesz do następnego zadania, należy zatrzymać poprzednią komórkę klikając czarny kwadrat w menu.
# 
# ![](../images/stop_cell.png)

# ## Zapisz model
# Również możesz samodzielnie zapisywać wagi modelu, a następnie możesz je wczytać (np. na serwerze produkcyjnym). Robi się to bardzo prosto.

# In[63]:


model.save_weights('../output/my_model_with_some_metadata.h5')


# Wczytywanie wag modelu.

# In[64]:


model.load_weights('../output/my_model_with_some_metadata.h5')


# ## Zadanie 8.4.1
# 
# Mając informację o `callback`’ach, spróbuj wytrenować model (dowolne zadanie, które robiliśmy przed tym) i zapisać wagi. Innymi słowy celem tego zadania jest użycie `callbacks` i zapisanie modelu do folderu `output`. Będzie nam potrzebny w następnym zadaniu.
# 
# Poświęć na to zadanie maksymalnie 1-2h (w tym czasie gdy trenuje się model, możesz sprawdzić przydatne linki, które są na dole).

# In[65]:


import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPool2D
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical

import mnist_reader

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt


# In[52]:


X_train, y_train = mnist_reader.load_mnist('../input/fashion', kind='train')
X_test, y_test = mnist_reader.load_mnist('../input/fashion', kind='t10k')

print(X_train.shape, X_test.shape)


# In[53]:


img_rows, img_cols = 28, 28

if K.image_data_format() == 'channels_last':
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)
else:
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
    
assert(X_train.shape == (60000, 28, 28, 1))    
assert(X_test.shape == (10000, 28, 28, 1)) 


# In[54]:


# normalizacja danych
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

X_train /= 255
X_test /= 255

num_classes = 10

# one-hot encoding dla zmiennej docelowej
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)


# In[55]:


def get_double_cnn_dropout():
    return Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25), 

        Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same',),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same',),
        Conv2D(128, kernel_size=(3, 3), activation='relu'),
        MaxPool2D(pool_size=(2, 2)),
        Dropout(0.25),

        Flatten(), # spłaszczanie danych, aby połączyć warstwy konwolucyjne z fully connected layers

        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

model = get_double_cnn_dropout()
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

model.summary()


# In[56]:


LOG_DIR = "../logs/tensorboard"
get_ipython().system('mkdir -p "$LOG_DIR"')


# In[57]:


callbacks = [ 
    EarlyStopping(patience=4), #jeśli 4 epoki z rzędu nie ma poprawy, to zatrzymaj się
    ModelCheckpoint('../output/model.best.hdf5', save_best_only=True), # zapis modelu po każdej epoce
    ReduceLROnPlateau(patience=3), #jeśli 3 epoki z rzędu nie ma poprawy, zmniejsz krok (learning_rate)
    TensorBoard(log_dir=LOG_DIR), # odkładanie logów, aby można było użyć Tensorboard
]


# In[58]:


history = model.fit(X_train, y_train,
          batch_size=512,
          epochs=5,
          verbose=1,
          validation_data=(X_test, y_test))


# In[59]:


score = model.evaluate(X_test, y_test, verbose=0)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")

print(f"CNN Error: {100-score[1]*100:.2f}%")


# In[60]:


model.save_weights('../output/my_model_sk.h5')


# In[61]:


model.load_weights('../output/my_model_sk.h5')


# ### 🤝🗣️ Współpraca 💪 i komunikacja 💬
# 
# - 👉 [#pml_module8](https://practicalmlcourse.slack.com/archives/C045CQBLK89) - to jest miejsce, gdzie można szukać pomocy i dzielić się doświadczeniem, także pomagać innym 🥰. 
# 
# Jeśli masz **pytanie z modułu 8**, to staraj się jak najdokładniej je sprecyzować, najlepiej wrzuć screen z twoim kodem i błędem, który się pojawił ✔️
# 
# - 👉 [#pml_module8_done](https://practicalmlcourse.slack.com/archives/C045CQCRAKT) - to miejsce, gdzie możesz dzielić się swoimi przerobionymi zadaniami, wystarczy, że wrzucisz screen z #done i numerem lekcji np. *#8.3.3_done*, śmiało dodaj komentarz, jeśli czujesz taką potrzebę, a także rozmawiaj z innymi o ich rozwiązaniach 😊 
# 
# - 👉 [#pml_module8_ideas](https://practicalmlcourse.slack.com/archives/C045CQE1B9P)- tutaj możesz dzielić się swoimi pomysłami związanymi z materiałem z tego modułu. 

# ## TensorBoard + CatBoost  ❤️
# Mówiąc o `TensorBoard` warto przypomnieć algorytm `CatBoost`, który również potrafi zintegrować się z `TensorBoard` i również posiada swój własny “dashboard”. Sprawdźmy to!

# In[ ]:


X, y = make_regression(n_samples=10000, n_features=100, random_state=2019)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)


# Tworzymy katalog, w którym zapiszemy wszystkie informacje związane z trenowaniem.

# In[ ]:


TRAIN_DIR="../logs/catboost"
get_ipython().system('mkdir -p "$TRAIN_DIR"')


# Trenujemy model.

# In[ ]:


model = ctb.CatBoostRegressor(train_dir=TRAIN_DIR)

model.fit(X_train, y_train, eval_set=(X_test, y_test), logging_level='Silent')
y_pred = model.predict(X_test)


# Uruchamiamy `TensorBoard`.

# In[ ]:


get_ipython().system('tensorboard --logdir=../logs/catboost --port=8050 --host=0.0.0.0')


# #  🚨🚨🚨  Następnie kliknij 👉👉👉 [tutaj](/hub/user-redirect/proxy/8050/) 👈👈👈

# ## TensorFlow Serving
# 
# <img src="../images/tf_arch.png" />
# 
# 
# To jest narzędzie, które zostało przygotowane przez Google do wdrażania modeli na produkcję. W uproszczeniu można powiedzieć, że TF Serving martwi się za nas, w jaki sposób efektywnie wykonywać żądania użytkowników (również potrafi je grupować), jak podmieniać nowe wersje modeli w niezauważalny dla użytkownika sposób i wiele innych rzeczy. Więcej można zobaczyć np. w tym [video](https://bit.ly/2S0n3PL).
# 
# Jednak `TF Serving` jest dość wymagający i dość ciężko jest go zainstalować np. w ramach kursu. Właściwie to jest jedna z największych trudności, którą można napotkać. Dlaczego są trudności? Np. `TF Serving` ma trudności [ze wsparciem python3](https://bit.ly/3bxRrIu).
# 
# Warto wiedzieć, że `TF Serving` istnieje i jeśli tak zdarzy się, że będziesz budować modele w `keras/tensorflow`, które trzeba będzie dostarczyć na produkcję, to jest to pierwsze narzędzie, o którym należy pomyśleć i sprawdzić.

# ## Przydatne linki:
# - [Serving Models in Production with TensorFlow Serving](https://bit.ly/2S0n3PL)
# - [How Zendesk Serves TensorFlow Models in Production](https://bit.ly/3uR15NO)
# - [Integrating Keras & TensorFlow: The Keras workflow, expanded](https://bit.ly/3tXKiHS)
# - [How to deploy Machine Learning models with TensorFlow](https://bit.ly/3hr07UN)
# - [Accelerating the Machine Learning Lifecycle with MLflow](https://bit.ly/3hvcdvW)
# - [The What-If Tool: Code-Free Probing of Machine Learning Models](https://bit.ly/3fqkxur)
# - [How to Build Flexible, Portable ML Stacks with Kubeflow and Elastifile (Next Rewind '18)](https://bit.ly/3eSR39z)
# - [Tensorflow in Docker on Kubernetes - Some Pitfalls](https://bit.ly/3eOc1pZ)
# - [Deploying Keras Deep Learning Models with Java](https://bit.ly/2Ql5wkW)
# - [Anaconda, Jupyter Notebook, TensorFlow and Keras for Deep Learning](https://bit.ly/3butzpc)
# 
