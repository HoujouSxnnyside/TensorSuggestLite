import tensorflow as tf

# Cargar modelo entrenado
model = tf.keras.models.load_model('text_classifier.keras')

# Convertir a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Opcional: optimización para reducir el tamaño del modelo
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Generar el modelo TFLite
tflite_model = converter.convert()

# Guardar modelo TFLite
with open('text_classifier.tflite', 'wb') as f:
    f.write(tflite_model)

print("Modelo convertido y guardado como 'text_classifier.tflite'")