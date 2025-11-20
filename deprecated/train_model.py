import json
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
import numpy as np

# Cargar datos desde el archivo JSON
with open('../question_data.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Preparar preguntas y categorías
preguntas = []
categorias = []

for categoria, contenido in data.items():
    # Agregar las respuestas como preguntas
    for respuesta in contenido['respuestas']:
        preguntas.append(respuesta['respuesta'])
        categorias.append(categoria)

    # Agregar los sinónimos como preguntas
    for sinonimo in contenido.get('sinonimos', []):
        preguntas.append(sinonimo)
        categorias.append(categoria)

# Preprocesar texto
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(preguntas)
sequences = tokenizer.texts_to_sequences(preguntas)
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences)
padded_sequences = padded_sequences / len(tokenizer.word_index)

# Codificar categorías
label_encoder = {label: idx for idx, label in enumerate(set(categorias))}
encoded_labels = np.array([label_encoder[label] for label in categorias])

# Dividir datos
X_train, X_test, y_train, y_test = train_test_split(
    padded_sequences, encoded_labels, test_size=0.1, random_state=42
)

# Crear modelo
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=32),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(len(label_encoder), activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Calcular pesos de clase
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(categorias),
    y=categorias
)

# Convertir a diccionario para usar en el entrenamiento
class_weights_dict = {label_encoder[label]: weight for label, weight in zip(np.unique(categorias), class_weights)}

# Entrenar el modelo con los pesos de clase
model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=30,
    batch_size=32,
    class_weight=class_weights_dict
)

# Guardar modelo como TensorFlow Lite
model.save('text_classifier.keras')

# Guardar el tokenizer en un archivo JSON
tokenizer_json = tokenizer.to_json()
with open('tokenizer.json', 'w', encoding='utf-8') as f:
    f.write(tokenizer_json)

# Guardar el mapeo de categorías
print("Label encoder mapping:", label_encoder)
with open('label_encoder.json', mode='w', encoding='utf-8') as f:
    json.dump(label_encoder, f, ensure_ascii=False)

print("Modelo y tokenizer guardados correctamente.")