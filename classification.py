import tensorflow as tf
import keras
from keras import layers
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
import seaborn as sns

pr_dir = 'output'
batch_size = 10

target_dset = pd.read_csv(os.path.join(pr_dir, 'targets.csv'))
feature_dset = pd.read_csv(os.path.join(pr_dir, 'features.csv'))

feature_dset['AQI'] = target_dset['AQI Category'].copy()
feature_dset.dropna(inplace=True)
dataset = feature_dset.sample(frac=1, random_state=0)
X = dataset.drop(columns=['location', 'AQI'])
Y = dataset['AQI']
n_labels = len(Y.unique())

l_encoder = LabelEncoder()
L = l_encoder.fit_transform(Y.values)
Labels = l_encoder.classes_
n_train = int(X.shape[0] * 0.85)
X_train, X_valid = tuple(np.split(X.to_numpy(), [n_train]))
Y_train, Y_valid = tuple(np.split(L, [n_train]))
print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape)
#
# # # # ------------------------------------------------------------------------------------
# #
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
train_dataset = train_dataset.batch(batch_size)
val_dataset = tf.data.Dataset.from_tensor_slices((X_valid, Y_valid))
val_dataset = val_dataset.batch(batch_size)
#
mean = np.mean(X.to_numpy())
var = np.var(X.to_numpy())

model = keras.Sequential()
model.add(layers.Input(shape=X_train.shape[1]))
model.add(layers.Normalization(mean=mean, variance=var))
model.add(layers.Dense(units=200, activation='relu'))
model.add(layers.Dense(units=20, activation='relu'))
model.add(layers.Dense(units=n_labels, activation='softmax'))

print(model.summary())
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])


EPOCH = 100
trained_model = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCH,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2)
)

metric = trained_model.history
plt.plot(trained_model.epoch, metric['loss'], metric['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

y_pred = model.predict(X.to_numpy())
y_pred = tf.argmax(y_pred, axis=1)

confusion_mtx = tf.math.confusion_matrix(L, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(confusion_mtx,
            xticklabels=Labels,
            yticklabels=Labels,
            annot=True, fmt='g')
plt.xlabel('Prediction')
plt.ylabel('Label')
plt.show()

y_true = L
d = np.where(np.abs(y_true - y_pred) == 2)[0]
print(dataset.iloc[d])
#
# plt.plot(Y_valid)
# Y_pred = model.predict(X_valid)
# plt.plot(Y_pred)
# plt.show()
#



