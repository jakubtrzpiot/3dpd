import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# 1. Load and preprocess data
train_ds = tf.keras.utils.image_dataset_from_directory(
    "data",
    labels="inferred",
    label_mode="binary",
    image_size=(64, 64),
    batch_size=32,
    validation_split=0.2,
    subset="training",
    seed=42
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "data",
    labels="inferred",
    label_mode="binary",
    image_size=(64, 64),
    batch_size=32,
    validation_split=0.2,
    subset="validation",
    seed=42
)

train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

# 2. Define a simple CNN model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

# 3. Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10  # Adjust as needed
)

# 4. Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('./model/model_quant.tflite', 'wb') as f:
    f.write(tflite_model)

print('Model trained and saved as model_quant.tflite')
