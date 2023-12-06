import tensorflow as tf

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

from sklearn.datasets import make_blobs
from matplotlib import pyplot
from pandas import DataFrame

# generate 2d classification dataset
X, y = make_blobs(n_samples=10000, centers=2, n_features=2, random_state = 444)
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:,0], y=X[:,1], label=y))
colors = {0:'red', 1:'blue', 2:'green'}
fig, ax = pyplot.subplots()
grouped = df.groupby('label')
for key, group in grouped:
    group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key])
pyplot.show()

df.to_csv("Loss Curves/Experimental_Dataset.csv", sep=';', decimal='.', index=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=666)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Define perfect model

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

number_epochs = 50

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs = number_epochs, batch_size = 64, validation_split = 0.2)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.show()

# Overfitted model 

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

number_epochs = 200

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs = number_epochs, batch_size = 64, validation_split = 0.1)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.show()

# Underfitted model

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(2,)),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(1000, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

number_epochs = 50

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs = number_epochs, batch_size = 64, validation_split = 0.2)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy}')

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')

plt.show()

import pandas as pd

# save data

accuracy_data = {
    'Epoch': list(range(1, (number_epochs + 1))),
    'Accuracy': history.history['accuracy'],
    'Type': ['Train'] * number_epochs
}

loss_data = {
    'Epoch': list(range(1, (number_epochs + 1))),
    'Loss': history.history['loss'],
    'Type': ['Train'] * number_epochs
}

val_accuracy_data = {
    'Epoch': list(range(1, (number_epochs + 1))),
    'Accuracy': history.history['val_accuracy'],
    'Type': ['Validation'] * number_epochs
}

val_loss_data = {
    'Epoch': list(range(1, (number_epochs + 1))),
    'Loss': history.history['val_loss'],
    'Type': ['Validation'] * number_epochs
}

df_accuracy = pd.DataFrame(accuracy_data)
df_loss = pd.DataFrame(loss_data)
df_val_accuracy = pd.DataFrame(val_accuracy_data)
df_val_loss = pd.DataFrame(val_loss_data)

df_accuracy.to_csv('Loss Curves/Acc_DS_Train_Underfit.csv', sep=';', decimal='.', index=False, header=False)
df_loss.to_csv('Loss Curves/Loss_DS_Train_Underfit.csv', sep=';', decimal='.', index=False, header=False)
df_val_accuracy.to_csv('Loss Curves/Acc_DS_Val_Underfit.csv', sep=';', decimal='.', index=False, header=False)
df_val_loss.to_csv('Loss Curves/Loss_DS_Val_Underfit.csv', sep=';', decimal='.', index=False, header=False)