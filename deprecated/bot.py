import tensorflow as tf
import numpy as np
import json
import unicodedata
from ai_edge_litert.interpreter import Interpreter

# Cargar el modelo LiteRT
model_interpreter = Interpreter(model_path='text_classifier.tflite')
model_interpreter.allocate_tensors()

# Obtener detalles de entrada y salida
input_details = model_interpreter.get_input_details()
output_details = model_interpreter.get_output_details()

# Cargar el tokenizer usado durante el entrenamiento
with open('tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer_config = json.load(f)
tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(tokenizer_config))

# Cargar las respuestas predefinidas desde el archivo JSON
with open('../question_data.json', 'r', encoding='utf-8') as f:
    question_data = json.load(f)

# Crear un diccionario de respuestas basado en label_encoder.json
with open('label_encoder.json', 'r', encoding='utf-8') as f:
    label_encoder = json.load(f)

# Invertir el mapeo de label_encoder para obtener índice -> categoría
respuestas = {v: k for k, v in label_encoder.items()}

# Depuración adicional
print("Diccionario de respuestas (índice -> categoría):", respuestas)

def normalizar_texto(texto):
    # Eliminar acentos y convertir a mayúsculas
    texto = unicodedata.normalize('NFD', texto)
    texto = ''.join(c for c in texto if unicodedata.category(c) != 'Mn')
    return texto.upper()

# Función para predecir
def responder(consulta):
    # Normalizar la consulta del usuario
    consulta = normalizar_texto(consulta)
    print(f"Consulta normalizada: {consulta}")

    # Buscar la categoría asociada en los sinónimos
    for categoria_key, contenido in question_data.items():
        categoria_normalizada = normalizar_texto(categoria_key)
        sinonimos = [normalizar_texto(s) for s in contenido.get('sinonimos', [])]
        if consulta == categoria_normalizada or consulta in sinonimos:
            consulta = categoria_key  # Usar la categoría original del JSON
            print(f"Consulta asociada a categoría: {consulta}")
            break

    # Preprocesar la pregunta
    sequence = tokenizer.texts_to_sequences([consulta])
    print(f"Secuencia tokenizada: {sequence}")

    input_data = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=input_details[0]['shape'][1])
    input_data = input_data.astype(np.float32)  # Convertir a FLOAT32
    # Depuración adicional
    print(f"Entrada al modelo (input_data): {input_data}")

    # Realizar la predicción
    model_interpreter.set_tensor(input_details[0]['index'], input_data)
    model_interpreter.invoke()
    output_data = model_interpreter.get_tensor(output_details[0]['index'])

    # Obtener la categoría predicha
    categoria_idx = np.argmax(output_data)
    categoria_predicha = respuestas.get(categoria_idx, "Desconocido")
    respuestas_categoria_local = question_data.get(categoria_predicha, [])
    # Depuración adicional
    print(f"Índice predicho: {categoria_idx}")
    print(f"Categoría predicha: {categoria_predicha}")

    return categoria_predicha, respuestas_categoria_local

# Ejemplo de uso
pregunta = "Traslado"
categoria, respuestas_categoria = responder(pregunta)

print(f"Categoría: {categoria}")
if isinstance(respuestas_categoria, dict) and 'respuestas' in respuestas_categoria:  # Verificar que sea un diccionario con la clave 'respuestas'
    for respuesta in respuestas_categoria['respuestas']:
        if isinstance(respuesta, dict) and 'respuesta' in respuesta:  # Verificar que sea un diccionario con la clave 'respuesta'
            print(f"- {respuesta['respuesta']}")
        else:
            print(f"Respuesta no válida: {respuesta}")
else:
    print("El formato de respuestas_categoria no es válido:", respuestas_categoria)
