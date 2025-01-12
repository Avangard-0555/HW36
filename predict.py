import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Загрузка модели
model = tf.keras.models.load_model('model/animal_classifier.h5')

# Классы
class_names = {0: 'Rabbit', 1: 'Dog'}

# Предсказание
def predict_image(image_path):
    image = load_img(image_path, target_size=(150, 150))
    img_array = img_to_array(image) / 255.0  # Нормализация
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = 1 if prediction[0] > 0.5 else 0
    confidence = prediction[0] if class_index == 1 else 1 - prediction[0]

    print(f"Класс: {class_names[class_index]}, Уверенность: {confidence[0]:.2f}")

# Тестовое изображение
predict_image('test_images/dog.jpg')  # Замените путь на ваше изображение








