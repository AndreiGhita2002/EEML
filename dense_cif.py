import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

dataset = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = dataset.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0

model = models.Sequential()
model.add(layers.Flatten(input_shape=(32, 32, 3)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dropout(0.1))
model.add(layers.Dense(10))

# print(model.weights)
model.summary()
print(model.input_shape)

# Compile and train the model:
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

EPOCHS = 20

history = model.fit(train_images, train_labels, epochs=EPOCHS,
                    validation_data=(test_images, test_labels))

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title("dense")
plt.show(block=True)

test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)

print(test_acc)
