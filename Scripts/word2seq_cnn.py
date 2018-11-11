# -*- coding: utf-8 -*-



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Embedding, Dropout, Flatten
from tensorflow.python.keras.layers import Conv1D, MaxPool1D
from tensorflow.python.keras import optimizers
from tensorflow.python.keras.preprocessing import text as keras_text, sequence as keras_seq
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from sklearn.metrics import confusion_matrix,accuracy_score,recall_score,precision_score,f1_score
from sklearn.utils import shuffle
from tensorflow import set_random_seed
import gc

#myrand=np.random.randint(1, 99999 + 1)
myrand=58584
np.random.seed(myrand)
set_random_seed(myrand)
z=0

EMBEDDING_SIZE=32
WORDS_SIZE=8000
INPUT_SIZE=700
NUM_CLASSES=2
EPOCHS=20

mydata = pd.read_csv('../Datasets/eRezeki/eRezeki_(text_class)_unclean.csv',header=0,encoding='utf-8')
mydata = mydata.loc[mydata['sentiment'] != "neutral"]
mydata['sentiment'].replace({"negative":0, "positive":1},inplace=True)

mydata1 = pd.read_csv('../Datasets/IMDB/all_random.csv',header=0,encoding='utf-8')
mydata = mydata.append(mydata1)
mydata = shuffle(mydata)

mydata1 = pd.read_csv('../Datasets/Amazon(sports_outdoors)/Amazon_UCSD.csv',header=0,encoding='utf-8')
mydata1['feedback'] = mydata1['feedback'].astype(str)
mydata = mydata.append(mydata1)
mydata = shuffle(mydata)

mydata1 = pd.read_csv('../Datasets/Yelp(zhang_paper)/yelp_zhang.csv',header=0,encoding='utf-8')
mydata1['feedback'] = mydata1['feedback'].astype(str)
mydata = mydata.append(mydata1)
del(mydata1)
gc.collect()
mydata = shuffle(mydata)

mydata = shuffle(mydata)
mydata = shuffle(mydata)

## Split Data
x_train, x_test, y_train, y_test = train_test_split(mydata.iloc[:,0], mydata.iloc[:,1],
                                                    test_size=0.3, 
                                                    random_state=myrand, 
                                                    shuffle=True)
old_y_test = y_test


## Create tokkenizer from full list of texts
tokenizer = keras_text.Tokenizer(char_level=False)
tokenizer.fit_on_texts(list(mydata['feedback']))
tokenizer.num_words=WORDS_SIZE

## Tokkenizing train data and create matrix
list_tokenized_train = tokenizer.texts_to_sequences(x_train)
x_train = keras_seq.pad_sequences(list_tokenized_train, 
                                  maxlen=INPUT_SIZE,
                                  padding='post')
x_train = x_train.astype(np.int64)

## Tokkenizing test data and create matrix
list_tokenized_test = tokenizer.texts_to_sequences(x_test)
x_test = keras_seq.pad_sequences(list_tokenized_test, 
                                 maxlen=INPUT_SIZE,
                                 padding='post')
x_test = x_test.astype(np.int64)

## OHE for labels
y_train = to_categorical(y_train, num_classes=NUM_CLASSES).astype(np.int64)
y_test = to_categorical(y_test, num_classes=NUM_CLASSES).astype(np.int64)


## Build CNN -----------------------------------------------------------------

model = Sequential(name='Word level cnn')

model.add(Embedding(input_dim =WORDS_SIZE,
                    output_dim=250,
                    input_length=INPUT_SIZE
                    ))


model.add(Conv1D(filters=250, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=50))
model.add(Conv1D(filters=250, kernel_size=2, padding='same', activation='relu'))
model.add(MaxPool1D(pool_size=10))
model.add(Flatten())
model.add(Dense(250, activation='relu'))

model.add(Dense(2, activation='softmax'))

## Define multiple optional optimizers
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
adam = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=1, decay=0.0, amsgrad=False)

## Compile model with metrics
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
print("Word-level CNN Conv1D model built: ")
model.summary()


## Training the CNN ----------------------------------------------------------

## Create TensorBoard callbacks

callbackdir= 'D:/ten'

tbCallback = TensorBoard(log_dir=callbackdir, 
                         histogram_freq=0, 
                         batch_size=128,
                         write_graph=True, 
                         write_grads=True, 
                         write_images=True)

tbCallback.set_model(model)


mld = 'D:/model/mcp_ALLL_word_conv1D.hdf5_%s'%(z)
## Create best model callback
mcp = ModelCheckpoint(filepath=mld, monitor="val_acc",
                      save_best_only=True, mode='max', period=1, verbose=1)

print('Training the CNN model')
model.fit(x = x_train,
          y = y_train,
          validation_data = (x_test, y_test),
          epochs = EPOCHS,
          batch_size = 128,
          verbose =2,
          callbacks=[mcp,tbCallback])


## Predict using the best-trained CNN model --------------------------------------
print('\nPredicting the model')
model = load_model(mld)
results = model.evaluate(x_test, y_test, batch_size=128)
for num in range(0,2):
    print(model.metrics_names[num]+': '+str(results[num]))


## Performance evaluation of the model --------------------------------------
    
print('\nConfusion Matrix')
predicted = model.predict_classes(x_test)
confusion = confusion_matrix(y_true=old_y_test, y_pred=predicted)
print(confusion)

## Performance measure
print('\nWeighted Accuracy: '+ str(accuracy_score(y_true=old_y_test, y_pred=predicted)))
print('Weighted precision: '+ str(precision_score(y_true=old_y_test, y_pred=predicted, average='weighted')))
print('Weighted recall: '+ str(recall_score(y_true=old_y_test, y_pred=predicted, average='weighted')))
print('Weighted f-measure: '+ str(f1_score(y_true=old_y_test, y_pred=predicted, average='weighted')))
