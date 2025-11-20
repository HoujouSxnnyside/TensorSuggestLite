import sys
import threading

from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QPushButton,
    QLineEdit,
    QHBoxLayout,
    QVBoxLayout,
    QFileDialog,
    QGraphicsDropShadowEffect,
    QProgressBar,
)

from json_interpreter import trainer


def _app_stylesheet() -> str:
    """Retorna una hoja de estilos QSS simple para simular neumorfismo.

    Nota: El verdadero neumorfismo se logra con degradados y sombras duales; aquí
    se aplica un estilo visual cercano que funciona con controles Qt estándar.
    """
    return """
    QWidget {
        background: qlineargradient(spread:pad, x1:0, y1:0, x2:1, y2:1,
                                    stop:0 #e6eef6, stop:1 #f7fbff);
        border-radius: 14px;
    }

    #titleLabel {
        color: #273142;
        margin-top: 6px;
        margin-bottom: 6px;
    }

    QLineEdit#pathField {
        background: #edf3f8;
        color: #2b3b46;
        border: none;
        border-radius: 12px;
        padding-left: 14px;
        padding-right: 14px;
    }

    QPushButton#neumoButton {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 #f9fbff, stop:1 #eaf2fb);
        color: #20323c;
        border: none;
        border-radius: 10px;
        padding: 8px 18px;
    }

    QPushButton#neumoButton:hover {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 #ffffff, stop:1 #e8f0fb);
    }

    QPushButton#neumoButton:pressed {
        background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                    stop:0 #dfeaf6, stop:1 #d0e4f4);
        padding-left: 20px; /* ligero movimiento al presionar */
    }
    """


class TensorSuggestLiteUI(QWidget):
    """Ventana principal con estilo "neumorphism" simple.

    Contiene:
    - Título en la parte superior: "TensorSuggestLite"
    - 3 botones para importar archivos
    - Un QLineEdit decorativo (solo lectura) para mostrar la ruta del archivo importado
    - Botones adicionales (Entrenar / Convertir a TFLite) y UX de progreso cuando hay un JSON seleccionado
    """

    training_progress = pyqtSignal(int)  # progreso desde 0 a 100

    def __init__(self):
        super().__init__()
        self.setWindowTitle("TensorSuggestLite")
        self.setFixedSize(520, 360)
        self.setStyleSheet(_app_stylesheet())

        self.selected_path = None

        self._build_ui()

    def _build_ui(self):
        # Tipografía
        title_font = QFont("Segoe UI", 18, QFont.Weight.Bold)
        btn_font = QFont("Segoe UI", 10)
        path_font = QFont("Segoe UI", 9)

        # Título
        title = QLabel("TensorSuggestLite")
        title.setFont(title_font)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setObjectName("titleLabel")

        # Botones principales de import
        self.btn_json = QPushButton("Importar JSON")
        self.btn_toml = QPushButton("Importar TOML")
        self.btn_yaml = QPushButton("Importar YAML")

        for b in (self.btn_json, self.btn_toml, self.btn_yaml):
            b.setFont(btn_font)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setFixedHeight(42)
            b.setObjectName("neumoButton")

            shadow = QGraphicsDropShadowEffect()
            shadow.setBlurRadius(14)
            shadow.setOffset(4, 4)
            shadow.setColor(Qt.GlobalColor.gray)
            b.setGraphicsEffect(shadow)

        self.btn_json.clicked.connect(lambda: self._import_file("JSON", "JSON files (*.json)"))
        self.btn_toml.clicked.connect(lambda: self._import_file("TOML", "TOML files (*.toml)"))
        self.btn_yaml.clicked.connect(lambda: self._import_file("YAML", "YAML files (*.yaml *.yml)"))

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)
        btn_layout.addWidget(self.btn_json)
        btn_layout.addWidget(self.btn_toml)
        btn_layout.addWidget(self.btn_yaml)

        # Campo de ruta (decorativo)
        self.path_field = QLineEdit()
        self.path_field.setReadOnly(True)
        self.path_field.setFont(path_font)
        self.path_field.setPlaceholderText("Ruta del archivo importado...")
        self.path_field.setObjectName("pathField")
        self.path_field.setFixedHeight(44)

        path_shadow = QGraphicsDropShadowEffect()
        path_shadow.setBlurRadius(20)
        path_shadow.setOffset(4, 4)
        path_shadow.setColor(Qt.GlobalColor.gray)
        self.path_field.setGraphicsEffect(path_shadow)

        # Botones y UX para acciones sobre JSON
        self.train_button = QPushButton("Entrenar modelo (JSON)")
        self.train_button.setFont(btn_font)
        self.train_button.setObjectName("neumoButton")
        self.train_button.setEnabled(False)
        self.train_button.clicked.connect(self._on_train_clicked)

        self.convert_button = QPushButton("Convertir a TFLite")
        self.convert_button.setFont(btn_font)
        self.convert_button.setObjectName("neumoButton")
        self.convert_button.setEnabled(False)
        self.convert_button.clicked.connect(self._on_convert_clicked)

        # Contador de items serializados y barra de progreso
        self.info_label = QLabel("")
        self.info_label.setFont(path_font)
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFixedHeight(18)

        actions_layout = QVBoxLayout()
        actions_layout.setSpacing(8)
        actions_layout.addWidget(self.train_button)
        actions_layout.addWidget(self.convert_button)
        actions_layout.addWidget(self.info_label)
        actions_layout.addWidget(self.progress)

        # Layout principal
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(18, 18, 18, 18)
        main_layout.setSpacing(14)
        main_layout.addWidget(title)
        main_layout.addLayout(btn_layout)
        main_layout.addWidget(self.path_field)
        main_layout.addLayout(actions_layout)

        self.setLayout(main_layout)

    def _import_file(self, kind: str, filter_str: str):
        dlg_title = f"Seleccionar archivo ({kind})"
        path, _ = QFileDialog.getOpenFileName(self, dlg_title, "", filter_str)
        if path:
            self.selected_path = path
            self.path_field.setText(path)

            # Solo habilitamos acciones para JSON por ahora
            if kind == "JSON":
                self.train_button.setEnabled(True)
                self.convert_button.setEnabled(False)  # habilitar después de entrenar
                self.info_label.setText("")
                self.progress.setValue(0)
            else:
                self.train_button.setEnabled(False)
                self.convert_button.setEnabled(False)

    def _on_train_clicked(self):
        if not self.selected_path:
            return

        self.train_button.setEnabled(False)
        self.info_label.setText("Iniciando entrenamiento...")
        self.progress.setValue(1)

        # Entrenamiento en hilo separado para no bloquear UI
        thread = threading.Thread(target=self._train_worker, daemon=True)
        thread.start()

    def _train_worker(self):
        try:
            # trainer.train_from_json debe actualizar progreso a través de callbacks
            def progress_cb(p):
                self._set_progress(p)

            result = trainer.train_from_json(self.selected_path, progress_cb)

            # result: dict con keys: items_serialized, tokenizer_path, model_path
            items = result.get("items_serialized", 0)
            tokenizer_path = result.get("tokenizer_path")
            model_path = result.get("model_path")

            # Planificar actualización de UI en hilo principal
            def _finish():
                self._set_progress(100)
                self.info_label.setText(f"Entrenamiento completado — items serializados: {items}")
                self.convert_button.setEnabled(True if model_path else False)

            QTimer.singleShot(0, _finish)
        except Exception as e:
            # Mostrar error en hilo principal
            QTimer.singleShot(0, lambda: self.info_label.setText(f"Error en entrenamiento: {e}"))
            self._set_progress(0)
        finally:
            # Rehabilitar botón en hilo principal
            QTimer.singleShot(0, lambda: self.train_button.setEnabled(True))

    def _set_progress(self, value: int):
        # Ejecutar en hilo principal
        QTimer.singleShot(0, lambda: self.progress.setValue(int(value)))

    def _on_convert_clicked(self):
        # Ejecutar conversión a TFLite en hilo aparte
        self.convert_button.setEnabled(False)
        self.info_label.setText("Convirtiendo a TFLite...")
        self.progress.setValue(1)

        thread = threading.Thread(target=self._convert_worker, daemon=True)
        thread.start()

    def _convert_worker(self):
        try:
            out_path = trainer.convert_to_tflite()
            QTimer.singleShot(0, lambda: self._set_progress(100))
            QTimer.singleShot(0, lambda: self.info_label.setText(f"Conversión completada: {out_path}"))
        except Exception as e:
            QTimer.singleShot(0, lambda: self.info_label.setText(f"Error al convertir: {e}"))
            self._set_progress(0)
        finally:
            QTimer.singleShot(0, lambda: self.convert_button.setEnabled(True))


def main():
    app = QApplication(sys.argv)
    win = TensorSuggestLiteUI()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
