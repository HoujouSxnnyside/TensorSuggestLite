import os
import json
from typing import Callable, Dict, Any, Optional


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def train_from_toml(toml_path: str, progress_cb: Optional[Callable[[int], None]] = None, log_cb: Optional[Callable[[str], None]] = None, epochs: int = 12) -> Dict[str, Any]:
    """Entrena un modelo a partir de un TOML con la estructura esperada.

    El flujo es equivalente al de JSON/YAML, salvo por la lectura del input
    (TOML). Los artefactos se guardan en `generated/toml/`.
    """
    try:
        import tensorflow as tf
        from sklearn.model_selection import train_test_split
        from sklearn.utils import class_weight
        import numpy as np
    except Exception as e:
        raise RuntimeError("TensorFlow y dependencias requeridas no están disponibles: %s" % e)

    try:
        import toml
    except Exception as e:
        raise RuntimeError("toml no está disponible: %s" % e)

    class _EpochProgressCallback(tf.keras.callbacks.Callback):
        def __init__(self, epochs: int, progress_cb: Optional[Callable[[int], None]] = None, log_cb: Optional[Callable[[str], None]] = None, start: int = 20, end: int = 85):
            super().__init__()
            self.epochs = epochs
            self.progress_cb = progress_cb
            self.log_cb = log_cb
            self.start = start
            self.end = end

        def on_epoch_end(self, epoch, logs=None):
            if not self.progress_cb:
                return
            try:
                pct = self.start + int(((epoch + 1) / float(self.epochs)) * (self.end - self.start))
                pct = max(0, min(100, pct))
                self.progress_cb(pct)
                if self.log_cb:
                    try:
                        self.log_cb(f'Epoch {epoch+1}/{self.epochs} completado ({pct}%).')
                    except Exception:
                        pass
            except Exception:
                pass

    def _safe_log(msg: str):
        if log_cb:
            try:
                log_cb(msg)
            except Exception:
                pass

    def _safe_progress(p: int):
        if progress_cb:
            try:
                progress_cb(p)
            except Exception:
                pass

    _safe_progress(0)
    _safe_log('Inicio del proceso de entrenamiento')

    with open(toml_path, 'r', encoding='utf-8') as f:
        data = toml.load(f)

    preguntas = []
    categorias = []

    for categoria, contenido in (data or {}).items():
        for respuesta in contenido.get('respuestas', []):
            if isinstance(respuesta, dict) and 'respuesta' in respuesta:
                preguntas.append(respuesta['respuesta'])
                categorias.append(categoria)
            elif isinstance(respuesta, str):
                preguntas.append(respuesta)
                categorias.append(categoria)
        for sinonimo in contenido.get('sinonimos', []):
            preguntas.append(sinonimo)
            categorias.append(categoria)

    if len(preguntas) == 0:
        raise ValueError('No se encontraron preguntas/entradas en el TOML proporcionado.')

    _safe_progress(5)
    _safe_log(f'Parseado TOML: {len(preguntas)} entradas preparadas, {len(set(categorias))} categorías detectadas')

    tokenizer = tf.keras.preprocessing.text.Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(preguntas)

    _safe_progress(10)
    _safe_log('Tokenizer entrenado con las entradas (fit_on_texts)')

    sequences = tokenizer.texts_to_sequences(preguntas)
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequences, padding='post')

    if len(tokenizer.word_index) > 0:
        padded = padded / float(len(tokenizer.word_index))

    unique_labels = list(sorted(set(categorias)))
    label_encoder = {label: idx for idx, label in enumerate(unique_labels)}
    encoded = [label_encoder[c] for c in categorias]
    encoded = np.array(encoded, dtype=np.int32)

    X_train, X_test, y_train, y_test = train_test_split(padded, encoded, test_size=0.1, random_state=42)

    _safe_progress(15)
    _safe_log('Preprocesamiento completado. Preparando dataset y definiendo modelo.')

    vocab_size = len(tokenizer.word_index) + 1
    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=64, mask_zero=True),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(len(unique_labels), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    try:
        classes = np.unique(categorias)
        cw = class_weight.compute_class_weight(class_weight='balanced', classes=classes, y=categorias)
        class_weights_dict = {label_encoder[label]: float(w) for label, w in zip(classes, cw)}
    except Exception:
        class_weights_dict = None

    cb = []
    if progress_cb or log_cb:
        cb.append(_EpochProgressCallback(epochs, progress_cb, log_cb=log_cb, start=20, end=85))
        try:
            def _lambda_on_epoch_end(epoch, logs, *, _epochs=epochs, _start=20, _end=85, _p=progress_cb, _l=log_cb):
                try:
                    pct = _start + int(((epoch + 1) / float(_epochs)) * (_end - _start))
                    pct = max(0, min(100, pct))
                    if _p:
                        try:
                            _p(pct)
                        except Exception:
                            pass
                    if _l:
                        try:
                            _l(f'Epoch {epoch+1}/{_epochs} completado ({pct}%).')
                        except Exception:
                            pass
                except Exception:
                    pass
            cb.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=_lambda_on_epoch_end))
        except Exception:
            pass

    model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        class_weight=class_weights_dict,
        callbacks=cb
    )

    root = _project_root()
    gen_dir = os.path.join(root, 'generated', 'toml')
    os.makedirs(gen_dir, exist_ok=True)

    tokenizer_path = os.path.join(gen_dir, 'tokenizer.json')
    model_path = os.path.join(gen_dir, 'text_classifier.keras')
    label_path = os.path.join(gen_dir, 'label_encoder.json')

    tok_json = tokenizer.to_json()
    with open(tokenizer_path, 'w', encoding='utf-8') as f:
        f.write(tok_json)
    _safe_log(f'Se ha creado el archivo tokenizer: {tokenizer_path}')

    model.save(model_path)
    _safe_log(f'Se ha creado el archivo Keras: {model_path}')

    with open(label_path, 'w', encoding='utf-8') as f:
        json.dump(label_encoder, f, ensure_ascii=False, indent=2)
    _safe_log(f'Se ha creado el archivo label encoder: {label_path}')

    _safe_progress(95)

    items_serialized = len(preguntas)

    _safe_progress(100)

    return {
        'items_serialized': items_serialized,
        'tokenizer_path': tokenizer_path,
        'model_path': model_path,
        'label_encoder_path': label_path,
    }


def convert_to_tflite(model_dir: Optional[str] = None, progress_cb: Optional[Callable[[int], None]] = None, log_cb: Optional[Callable[[str], None]] = None) -> str:
    try:
        import tensorflow as tf
    except Exception as e:
        raise RuntimeError("TensorFlow no está disponible: %s" % e)

    def _safe_log(msg: str):
        if log_cb:
            try:
                log_cb(msg)
            except Exception:
                pass

    def _safe_progress(p: int):
        if progress_cb:
            try:
                progress_cb(int(max(0, min(100, int(p)))))
            except Exception:
                pass

    _safe_progress(5)
    _safe_log('Iniciando conversión a TFLite')

    root = _project_root()
    if model_dir:
        model_path = os.path.join(model_dir, 'text_classifier.keras')
        out_dir = model_dir
    else:
        model_path = os.path.join(root, 'generated', 'toml', 'text_classifier.keras')
        out_dir = os.path.join(root, 'generated', 'toml')

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No se encontró el modelo Keras en: {model_path}")

    model = tf.keras.models.load_model(model_path)
    _safe_progress(25)
    _safe_log('Modelo cargado. Iniciando conversión con TFLiteConverter')
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    try:
        tflite_model = converter.convert()
    except Exception as e:
        _safe_log(f'Error durante la conversión: {e}')
        raise

    _safe_progress(90)
    _safe_log('Conversión completada (en memoria). Guardando archivo TFLite...')

    out_path = os.path.join(out_dir, 'text_classifier.tflite')
    with open(out_path, 'wb') as f:
        f.write(tflite_model)

    _safe_progress(100)
    _safe_log(f'TFLite guardado en: {out_path}')

    return out_path
