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
