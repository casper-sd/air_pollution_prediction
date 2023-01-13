import tensorflow as tf
import keras
from keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

pr_dir = 'output3'
batch_size = 10

target_dset = pd.read_csv(os.path.join(pr_dir, 'targets.csv'))
feature_dset = pd.read_csv(os.path.join(pr_dir, 'features.csv'))
dset = pd.merge(feature_dset, target_dset, on='location')
dset.dropna(inplace=True)
dataset = dset.sample(frac=1, random_state=0)
X = dataset.drop(columns=['location', 'AQI', 'AQI Category'])
Y = dataset['AQI']

n_train = int(X.shape[0] * 0.85)
X_train, X_valid = tuple(np.split(X.to_numpy(), [n_train]))
Y_train, Y_valid = tuple(np.split(Y.values, [n_train]))
print(X_train.shape, Y_train.shape, X_valid.shape, Y_valid.shape)

# # # ------------------------------------------------------------------------------------
#
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
model.add(layers.Dense(units=100, activation='elu'))
model.add(layers.Dense(units=50, activation='elu'))
model.add(layers.Dropout(rate=0.25))
model.add(layers.Dense(units=20, activation='elu'))
model.add(layers.Dense(units=5, activation='elu'))
model.add(layers.Dense(units=1))

print(model.summary())
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.MeanAbsoluteError(),
    metrics=[tf.keras.metrics.MeanAbsoluteError()]
)

EPOCH = 100
trained_model = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCH,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=8)
)
#
metric = trained_model.history
plt.plot(trained_model.epoch, metric['loss'], metric['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.show()

y_true = Y_valid
plt.plot(y_true)
y_pred = np.squeeze(model.predict(X_valid), axis=1)
plt.plot(y_pred)
plt.show()

# dataset['pred_aqi'] = y_pred
# dataset['true_aqi'] = y_true
# Outliers = []
#
# for i, loc in dataset.iterrows():
#     lo = loc['location'].split('-')
#     if abs(loc['pred_aqi'] - loc['true_aqi']) > 30:
#         Outliers.append({'State': lo[0], 'City': lo[1], 'Indicator': '+' if loc['pred_aqi'] < loc['true_aqi'] else '-',
#                          'Predicted AQI': loc['pred_aqi'], 'Actual AQI': loc['true_aqi']})
#
# pd.DataFrame(Outliers).to_csv(os.path.join(pr_dir, 'outliers.csv'), index=False)
