import os
import json
from typing import Callable, Dict, Any, Optional


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def train_from_json(json_path: str, progress_cb: Optional[Callable[[int], None]] = None, epochs: int = 12) -> Dict[str, Any]:
    """Entrena un modelo a partir de un JSON con la estructura esperada.

    Importa TensorFlow y scikit-learn en tiempo de ejecución para permitir que el
    módulo sea importado incluso si TensorFlow no está disponible.
    """
    try:
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
        from sklearn.utils import class_weight
        import numpy as np
    except Exception as e:
        raise RuntimeError("TensorFlow y dependencias requeridas no están disponibles: %s" % e)

    class _EpochProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, epochs: int, progress_cb: Optional[Callable[[int], None]] = None):
            super().__init__()
            self.epochs = epochs
            self.progress_cb = progress_cb

        def on_epoch_end(self, epoch, logs=None):
            if not self.progress_cb:
                return
            percent = int(((epoch + 1) / float(self.epochs)) * 90)
            try:
                self.progress_cb(percent)
            except Exception:
                pass

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    preguntas = []
    categorias = []

    # Intentar parsear la estructura conocida
    for categoria, contenido in data.items():
        # respuestas esperadas como lista de dicts con clave 'respuesta'
        for respuesta in contenido.get('respuestas', []):
            if isinstance(respuesta, dict) and 'respuesta' in respuesta:
                preguntas.append(respuesta['respuesta'])
                categorias.append(categoria)
            elif isinstance(respuesta, str):
                preguntas.append(respuesta)
                categorias.append(categoria)

        # sinonimos pueden ser lista de strings
        for sinonimo in contenido.get('sinonimos', []):
            preguntas.append(sinonimo)
            categorias.append(categoria)

    if len(preguntas) == 0:
        raise ValueError('No se encontraron preguntas/entradas en el JSON proporcionado.')

    # Tokenizer y secuencias
    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(preguntas)
    sequences = tokenizer.texts_to_sequences(preguntas)
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

    # Normalizar (opcional)
    if len(tokenizer.word_index) > 0:
        padded = padded / float(len(tokenizer.word_index))

    # Codificar categorías
    unique_labels = list(sorted(set(categorias)))
    label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = [label_encoder[c] for c in categorias]
    encoded = np.array(encoded, dtype=np.int32)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(padded, encoded, test_size=0.1, random_state=42)

    # Modelo simple
    vocab_size = len(tokenizer.word_index) + 1
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, mask_zero=True),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(unique_labels), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Pesos de clase (opcional, robustez)
    try:
        classes = np.unique(categorias)
        cw = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=categorias)
        class_weights_dict = {label_encoder[label]: float(w) for label, w in zip(classes, cw)}
    except Exception:
        class_weights_dict = None

    # Callbacks para progreso
    cb = []
    if progress_cb:
        cb.append(_EpochProgressCallback(epochs, progress_cb))

    # Fit
    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        class_weight=class_weights_dict,
        callbacks=cb
    )

    # Guardar artefactos en el root del proyecto
    root = _project_root()
    tokenizer_path = os.path.join(root, 'tokenizer.json')
    model_path = os.path.join(root, 'text_classifier.keras')
    label_path = os.path.join(root, 'label_encoder.json')

    # tokenizer
    tok_json = tokenizer.to_json()
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        f.write(tok_json)

    # model
    model.save(model_path)

    # label encoder
    with open(label_path, 'w', encoding='utf-8') as f:
        json.dump(label_encoder, f, ensure_ascii=False, indent=2)

    # Informe final y progresos
    if progress_cb:
        try:
            progress_cb(95)
        except Exception:
            pass

    items_serialized = len(preguntas)

    if progress_cb:
        try:
            progress_cb(100)
        except Exception:
            pass

    return {
        'items_serialized': items_serialized,
        'tokenizer_path': tokenizer_path,
        'model_path': model_path,
        'label_encoder_path': label_path,
    }


def convert_to_tflite() -> str:
    """Convierte el modelo Keras guardado (text_classifier.keras) a TFLite.

    Busca en el directorio raíz del proyecto y escribe `text_classifier.tflite`.
    Retorna la ruta del archivo TFLite.
    """
    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError("TensorFlow no está disponible: %s" % e)

    root = _project_root()
    model_path = os.path.join(root, 'text_classifier.keras')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo Keras en: {model_path}")

    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    out_path = os.path.join(root, 'text_classifier.tflite')
    with open(out_path, 'wb') as f:
        f.write(tflite_model)

    return out_path
