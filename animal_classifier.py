pip install -r requirements.txt


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
import os

# Установки
DATASET_DIR = 'dataset'
BATCH_SIZE = 32
IMG_SIZE = (150, 150)

# Проверяем, есть ли датасет
if not os.path.exists(DATASET_DIR):
    raise FileNotFoundError(f"Папка {DATASET_DIR} не найдена. Убедитесь, что изображения находятся в правильной директории.")

# Генераторы данных
datagen_train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # 20% данных для валидации
)

train_generator = datagen_train.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',  # Два класса
    subset='training'
)

validation_generator = datagen_train.flow_from_directory(
    DATASET_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation'
)

# Модель
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')  # Сигмоида для двух классов
])

# Компиляция модели
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Обучение
history = model.fit(
    train_generator,
    epochs=10,  # Количество эпох обучения
    validation_data=validation_generator
)

# Сохранение модели
if not os.path.exists('model'):
    os.makedirs('model')
model.save('model/animal_classifier.h5')
print("Модель сохранена в 'model/animal_classifier.h5'.")

# Построение графиков
plt.plot(history.history['accuracy'], label='Точность на обучении')
plt.plot(history.history['val_accuracy'], label='Точность на валидации')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()
plt.show()

