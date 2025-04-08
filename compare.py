import json

# Cargar question_data
with open('question_data.json', 'r', encoding='utf-8') as f:
    question_data = json.load(f)

# Cargar label_encoder
with open('label_encoder.json', 'r', encoding='utf-8') as f:
    label_encoder = json.load(f)

# Obtener las claves de ambos
question_data_keys = set(question_data.keys())
label_encoder_keys = set(label_encoder.keys())

# Comparar las claves
faltantes_en_label_encoder = question_data_keys - label_encoder_keys
faltantes_en_question_data = label_encoder_keys - question_data_keys

print("Claves faltantes en label_encoder:", faltantes_en_label_encoder)
print("Claves faltantes en question_data:", faltantes_en_question_data)

# Verificar si coinciden completamente
if not faltantes_en_label_encoder and not faltantes_en_question_data:
    print("Las claves de question_data y label_encoder coinciden.")
else:
    print("Hay discrepancias entre question_data y label_encoder.")