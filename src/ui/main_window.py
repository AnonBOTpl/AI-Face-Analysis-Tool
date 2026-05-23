import cv2
import os
import json
import csv
import logging
import threading
import numpy as np
from pathlib import Path
from datetime import datetime

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QGroupBox, QComboBox, QCheckBox,
    QPushButton, QLabel, QProgressBar, QTableWidget,
    QTableWidgetItem, QFileDialog, QMessageBox,
    QSplitter, QScrollArea, QFrame, QHeaderView,
    QStatusBar, QGridLayout,
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QSize, QRect
from PyQt6.QtGui import QPixmap, QImage, QDragEnterEvent, QDropEvent, QPainter, QFont

from src.analyzer.pipeline import AnalysisPipeline
from src.analyzer.age_gender import AGE_GENDER_MODELS
from src.ui.styles import DARK_THEME
from src.utils.i18n import _, set_language, get_language
from src.utils.image import draw_results, draw_boxes_preview, numpy_to_pixmap
from src.config import (
    APP_NAME, APP_VERSION, DETECTOR_BACKENDS, GPU_NAME,
    EMOTION_LABELS_PL, DEFAULT_AGE_GENDER_MODEL,
)

logger = logging.getLogger(__name__)


class ImageLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._pixmap = None
        self._img_size = (0, 0)
        self._face_boxes = []

    def display(self, pixmap, img_w=0, img_h=0):
        self._pixmap = pixmap
        self._img_size = (img_w, img_h)
        self._update()

    def setFaceBoxes(self, boxes):
        self._face_boxes = boxes

    def _update(self):
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            super().setPixmap(scaled)

    def resizeEvent(self, event):
        self._update()
        super().resizeEvent(event)

    def mouseMoveEvent(self, event):
        if not self._face_boxes or not self._pixmap:
            self.setToolTip("")
            return
        pm = self.pixmap()
        if not pm:
            return
        x_off = (self.width() - pm.width()) // 2
        y_off = (self.height() - pm.height()) // 2
        mx = int(event.position().x() - x_off)
        my = int(event.position().y() - y_off)
        if mx < 0 or my < 0 or mx > pm.width() or my > pm.height():
            self.setToolTip("")
            return
        img_w, img_h = self._img_size
        if img_w == 0 or img_h == 0:
            return
        sx = img_w / pm.width()
        sy = img_h / pm.height()
        ox = mx * sx
        oy = my * sy
        for x1, y1, x2, y2, tip in self._face_boxes:
            if x1 <= ox <= x2 and y1 <= oy <= y2:
                self.setToolTip(tip)
                return
        self.setToolTip("")


class AnalysisWorker(QThread):
    finished = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, pipeline, image):
        super().__init__()
        self.pipeline = pipeline
        self.image = image

    def run(self):
        try:
            self.progress.emit(_("analyzing"))
            results = self.pipeline.analyze(self.image)
            self.finished.emit(results)
        except Exception as e:
            logger.exception("Analysis failed")
            self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1400, 900)
        self.resize(1600, 1000)
        self.setAcceptDrops(True)

        self.current_image = None
        self.current_results = []
        self.annotated_image = None
        self.image_path = None
        self.webcam_timer = QTimer()
        self.webcam_timer.timeout.connect(self._webcam_frame)
        self.webcam_capture = None
        self.pipeline = None
        self.webcam_pipeline = None
        self.worker = None

        self._setup_ui()
        self._apply_theme()
        self._update_ag_model_desc()
        self._setup_pipeline()
        self.statusBar().showMessage(_("ready"))

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        self.tabs.addTab(self._build_single_tab(), _("tab_single"))
        self.tabs.addTab(self._build_batch_tab(), _("tab_batch"))
        self.tabs.addTab(self._build_webcam_tab(), _("tab_webcam"))
        self.statusBar().setStyleSheet("QStatusBar { font-size: 12px; }")

    def _apply_theme(self):
        self.setStyleSheet(DARK_THEME)

    def _build_single_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(splitter)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.image_label = ImageLabel(_("drag_drop"))
        self.image_label.setObjectName("imageLabel")
        self.image_label.setMinimumSize(500, 400)
        left_layout.addWidget(self.image_label)
        self.img_info_label = QLabel()
        self.img_info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.img_info_label.setStyleSheet("color: #6c7086; font-size: 11px;")
        left_layout.addWidget(self.img_info_label)
        left_scroll.setWidget(left_widget)
        splitter.addWidget(left_scroll)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)
        settings_group = QGroupBox(_("settings"))
        settings_grid = QGridLayout(settings_group)
        settings_grid.addWidget(QLabel(_("detector")), 0, 0)
        self.detector_combo = QComboBox()
        for name in DETECTOR_BACKENDS:
            self.detector_combo.addItem(name)
        settings_grid.addWidget(self.detector_combo, 0, 1)
        settings_grid.addWidget(QLabel(_("age_gender_model")), 1, 0)
        self.ag_model_combo = QComboBox()
        self.ag_model_keys = list(AGE_GENDER_MODELS.keys())
        self.ag_model_names = [m["name"] for m in AGE_GENDER_MODELS.values()]
        self.ag_model_combo.addItems(self.ag_model_names)
        self.ag_model_combo.setCurrentIndex(self.ag_model_keys.index(DEFAULT_AGE_GENDER_MODEL))
        self.ag_model_combo.currentIndexChanged.connect(self._update_ag_model_desc)
        settings_grid.addWidget(self.ag_model_combo, 1, 1)
        self.ag_model_desc = QLabel()
        self.ag_model_desc.setStyleSheet("color: #6c7086; font-size: 10px;")
        self.ag_model_desc.setWordWrap(True)
        settings_grid.addWidget(self.ag_model_desc, 2, 0, 1, 2)
        settings_grid.addWidget(QLabel(_("actions")), 3, 0)
        actions_widget = QWidget()
        actions_layout = QHBoxLayout(actions_widget)
        actions_layout.setContentsMargins(0, 0, 0, 0)
        self.cb_age = QCheckBox(_("age"))
        self.cb_age.setChecked(True)
        self.cb_gender = QCheckBox(_("gender"))
        self.cb_gender.setChecked(True)
        self.cb_emotion = QCheckBox(_("emotion"))
        self.cb_emotion.setChecked(True)
        actions_layout.addWidget(self.cb_age)
        actions_layout.addWidget(self.cb_gender)
        actions_layout.addWidget(self.cb_emotion)
        settings_grid.addWidget(actions_widget, 3, 1)
        right_layout.addWidget(settings_group)
        self.btn_select = QPushButton(_("select_image"))
        self.btn_select.clicked.connect(self._select_image)
        right_layout.addWidget(self.btn_select)
        self.btn_analyze = QPushButton(_("analyze"))
        self.btn_analyze.setObjectName("analyzeBtn")
        self.btn_analyze.clicked.connect(self._analyze)
        self.btn_analyze.setEnabled(False)
        right_layout.addWidget(self.btn_analyze)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        right_layout.addWidget(self.progress_bar)
        results_group = QGroupBox(_("results"))
        results_layout = QVBoxLayout(results_group)
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(5)
        self.results_table.setHorizontalHeaderLabels([
            "#", _("age"), _("gender"), _("emotion"), "Conf."
        ])
        self.results_table.horizontalHeader().setStretchLastSection(True)
        self.results_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.results_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.results_table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )
        results_layout.addWidget(self.results_table)
        btn_row = QHBoxLayout()
        self.btn_save_img = QPushButton(_("save_image"))
        self.btn_save_img.setObjectName("saveBtn")
        self.btn_save_img.clicked.connect(self._save_image)
        self.btn_save_img.setEnabled(False)
        btn_row.addWidget(self.btn_save_img)
        self.btn_save_csv = QPushButton(_("save_csv"))
        self.btn_save_csv.clicked.connect(self._save_csv)
        self.btn_save_csv.setEnabled(False)
        btn_row.addWidget(self.btn_save_csv)
        self.btn_save_json = QPushButton(_("save_json"))
        self.btn_save_json.clicked.connect(self._save_json)
        self.btn_save_json.setEnabled(False)
        btn_row.addWidget(self.btn_save_json)
        results_layout.addLayout(btn_row)
        right_layout.addWidget(results_group, stretch=1)
        splitter.addWidget(right_panel)
        splitter.setSizes([600, 400])
        return tab

    def _build_batch_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.addWidget(QLabel(_("tab_batch")))
        self.btn_batch_select = QPushButton(_("select_folder"))
        self.btn_batch_select.clicked.connect(self._select_batch_folder)
        layout.addWidget(self.btn_batch_select)
        self.batch_progress = QProgressBar()
        self.batch_progress.setRange(0, 100)
        self.batch_progress.hide()
        layout.addWidget(self.batch_progress)
        self.batch_table = QTableWidget()
        self.batch_table.setColumnCount(5)
        self.batch_table.setHorizontalHeaderLabels([
            _("face"), _("age"), _("gender"), _("emotion"), _("image_info")
        ])
        self.batch_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.batch_table, stretch=1)
        self.btn_batch_export = QPushButton(_("save_csv"))
        self.btn_batch_export.clicked.connect(self._export_batch_csv)
        self.btn_batch_export.setEnabled(False)
        layout.addWidget(self.btn_batch_export)
        return tab

    def _build_webcam_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        controls = QHBoxLayout()
        self.btn_webcam = QPushButton(_("start_camera"))
        self.btn_webcam.clicked.connect(self._toggle_webcam)
        controls.addWidget(self.btn_webcam)
        self.webcam_detector = QComboBox()
        for name in DETECTOR_BACKENDS:
            self.webcam_detector.addItem(name)
        controls.addWidget(QLabel(_("age_gender_model")))
        self.wc_ag_model = QComboBox()
        self.wc_ag_model.addItems(self.ag_model_names)
        self.wc_ag_model.setCurrentIndex(self.ag_model_keys.index(DEFAULT_AGE_GENDER_MODEL))
        controls.addWidget(self.wc_ag_model)
        controls.addWidget(QLabel(_("actions")))
        self.wc_age = QCheckBox(_("age"))
        self.wc_age.setChecked(True)
        self.wc_gender = QCheckBox(_("gender"))
        self.wc_gender.setChecked(True)
        self.wc_emotion = QCheckBox(_("emotion"))
        self.wc_emotion.setChecked(True)
        controls.addWidget(self.wc_age)
        controls.addWidget(self.wc_gender)
        controls.addWidget(self.wc_emotion)
        controls.addStretch()
        layout.addLayout(controls)
        self.webcam_label = QLabel(_("webcam_hint"))
        self.webcam_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.webcam_label.setObjectName("imageLabel")
        self.webcam_label.setMinimumSize(640, 480)
        layout.addWidget(self.webcam_label, stretch=1)
        self.webcam_status = QLabel()
        self.webcam_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.webcam_status.setStyleSheet("color: #6c7086;")
        layout.addWidget(self.webcam_status)
        return tab

    def _setup_pipeline(self):
        detector_key = self.detector_combo.currentText()
        backend = DETECTOR_BACKENDS.get(detector_key, "mediapipe")
        ag_model = self.ag_model_keys[self.ag_model_combo.currentIndex()]
        self.pipeline = AnalysisPipeline(
            detector_backend=backend,
            age_gender_model=ag_model,
            enable_age=self.cb_age.isChecked(),
            enable_gender=self.cb_gender.isChecked(),
            enable_emotion=self.cb_emotion.isChecked(),
        )

    def _update_ag_model_desc(self):
        idx = self.ag_model_combo.currentIndex()
        key = self.ag_model_keys[idx]
        self.ag_model_desc.setText(AGE_GENDER_MODELS[key]["description"])

    def _recreate_pipeline(self):
        idx = self.tabs.currentIndex()
        if idx == 0:
            detector_key = self.detector_combo.currentText()
            backend = DETECTOR_BACKENDS.get(detector_key, "mediapipe")
            ag_model = self.ag_model_keys[self.ag_model_combo.currentIndex()]
            self.pipeline = AnalysisPipeline(
                detector_backend=backend,
                age_gender_model=ag_model,
                enable_age=self.cb_age.isChecked(),
                enable_gender=self.cb_gender.isChecked(),
                enable_emotion=self.cb_emotion.isChecked(),
            )

    def _select_image(self):
        path, _filter = QFileDialog.getOpenFileName(
            self, _("select_image"), "",
            "Images (*.jpg *.jpeg *.png *.bmp *.webp *.tiff)"
        )
        if path:
            self._load_image(path)

    def _load_image(self, path):
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Failed to load image")
            self.current_image = img
            self.image_path = path
            h, w = img.shape[:2]
            pixmap = numpy_to_pixmap(img)
            self.image_label.display(pixmap, w, h)
            self.image_label.setFaceBoxes([])
            self.image_label.setText("")
            size_mb = os.path.getsize(path) / (1024 * 1024)
            name = os.path.basename(path)
            self.img_info_label.setText(f"{name} | {w}x{h} | {size_mb:.1f}MB")
            self.btn_analyze.setEnabled(True)
            self.current_results = []
            self.annotated_image = None
            self.results_table.setRowCount(0)
            self.btn_save_img.setEnabled(False)
            self.btn_save_csv.setEnabled(False)
            self.btn_save_json.setEnabled(False)
            self.statusBar().showMessage(f"Loaded: {name}")
        except Exception as e:
            QMessageBox.critical(self, _("error"), str(e))

    def _analyze(self):
        if self.current_image is None:
            return
        self._recreate_pipeline()
        self.btn_analyze.setEnabled(False)
        self.btn_select.setEnabled(False)
        self.progress_bar.show()
        self.results_table.setRowCount(0)
        self.worker = AnalysisWorker(self.pipeline, self.current_image)
        self.worker.finished.connect(self._on_analysis_done)
        self.worker.error.connect(self._on_analysis_error)
        self.worker.progress.connect(lambda msg: self.statusBar().showMessage(msg))
        self.worker.start()

    def _on_analysis_done(self, results):
        self.current_results = results
        lang = get_language()
        self.results_table.setRowCount(len(results))
        for i, res in enumerate(results):
            self.results_table.setItem(i, 0, QTableWidgetItem(str(i + 1)))
            self.results_table.setItem(i, 1, QTableWidgetItem(str(res.age) if res.age is not None else "-"))
            self.results_table.setItem(i, 2, QTableWidgetItem(str(res.gender) if res.gender else "-"))
            em = res.emotion or "-"
            if lang == "pl" and em in EMOTION_LABELS_PL:
                em = EMOTION_LABELS_PL[em]
            self.results_table.setItem(i, 3, QTableWidgetItem(em))
            self.results_table.setItem(i, 4, QTableWidgetItem(f"{res.confidence:.0%}"))
        if results:
            self.annotated_image = draw_results(self.current_image, results, lang)
            h, w = self.current_image.shape[:2]
            preview = draw_boxes_preview(self.current_image, results)
            pixmap = numpy_to_pixmap(preview)
            self.image_label.display(pixmap, w, h)
            face_boxes = []
            for res in results:
                x1, y1, x2, y2 = res.bbox
                parts = []
                if res.age is not None:
                    parts.append(f"Age: {res.age}")
                if res.gender:
                    g = res.gender
                    if res.gender_confidence:
                        g += f" ({res.gender_confidence:.0%})"
                    parts.append(g)
                em = res.emotion or ""
                if lang == "pl" and em in EMOTION_LABELS_PL:
                    em = EMOTION_LABELS_PL[em]
                if em:
                    parts.append(f"Emotion: {em}")
                parts.append(f"Conf: {res.confidence:.0%}")
                face_boxes.append((x1, y1, x2, y2, " | ".join(parts)))
            self.image_label.setFaceBoxes(face_boxes)
            self.btn_save_img.setEnabled(True)
            self.btn_save_csv.setEnabled(True)
            self.btn_save_json.setEnabled(True)
        else:
            self.image_label.setText(_("no_faces"))
            self.image_label.setFaceBoxes([])
        self.progress_bar.hide()
        self.btn_analyze.setEnabled(True)
        self.btn_select.setEnabled(True)
        if self.pipeline:
            self.pipeline.cleanup()
        self.statusBar().showMessage(
            f"{_('completed')} — {len(results)} {_('face').lower()}{'s' if len(results) != 1 else ''}"
        )

    def _on_analysis_error(self, msg):
        self.progress_bar.hide()
        self.btn_analyze.setEnabled(True)
        self.btn_select.setEnabled(True)
        QMessageBox.critical(self, _("error"), msg)

    def _save_image(self):
        if self.annotated_image is None:
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, _("save_image"), "result.png",
            "PNG (*.png);;JPEG (*.jpg *.jpeg);;BMP (*.bmp)"
        )
        if path:
            cv2.imwrite(path, self.annotated_image)
            self.statusBar().showMessage(f"Saved: {os.path.basename(path)}")

    def _save_csv(self):
        self._export_table_csv(self.current_results, single=True)

    def _save_json(self):
        if not self.current_results:
            return
        path, _filter = QFileDialog.getSaveFileName(
            self, _("save_json"), "results.json",
            "JSON (*.json)"
        )
        if not path:
            return
        data = []
        for res in self.current_results:
            item = {
                "age": res.age,
                "gender": res.gender,
                "gender_confidence": res.gender_confidence,
                "emotion": res.emotion,
                "emotions": res.emotions,
                "confidence": res.confidence,
                "bbox": list(res.bbox),
            }
            if self.image_path:
                item["image"] = self.image_path
            data.append(item)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        self.statusBar().showMessage(f"Exported: {os.path.basename(path)}")

    def _export_table_csv(self, results, single=False):
        if not results:
            return
        if single:
            path, _filter = QFileDialog.getSaveFileName(
                self, _("save_csv"), "results.csv",
                "CSV (*.csv)"
            )
        else:
            return
        if not path:
            return
        lang = get_language()
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["Face", "Age", "Gender", "Gender_Confidence",
                             "Emotion", "Confidence"])
            for i, res in enumerate(results):
                em = res.emotion or ""
                if lang == "pl":
                    em = EMOTION_LABELS_PL.get(em, em)
                writer.writerow([
                    i + 1, res.age, res.gender, res.gender_confidence,
                    em, f"{res.confidence:.3f}"
                ])
        self.statusBar().showMessage(f"Exported: {os.path.basename(path)}")

    def _select_batch_folder(self):
        folder = QFileDialog.getExistingDirectory(self, _("select_folder"))
        if folder:
            self._run_batch(folder)

    def _run_batch(self, folder):
        self._recreate_pipeline()
        self.batch_progress.setValue(0)
        self.batch_progress.show()
        self.btn_batch_select.setEnabled(False)
        self.batch_table.setRowCount(0)
        extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        files = [p for p in Path(folder).iterdir()
                 if p.suffix.lower() in extensions]
        lang = get_language()
        all_results = []
        for idx, filepath in enumerate(files):
            self.statusBar().showMessage(
                _("batch_progress").format(current=idx + 1, total=len(files))
            )
            self.batch_progress.setValue(int((idx + 1) / len(files) * 100))
            QThread.msleep(1)
            img = cv2.imdecode(np.fromfile(str(filepath), dtype=np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            try:
                results = self.pipeline.analyze(img)
            except Exception:
                continue
            for res in results:
                em = res.emotion or ""
                if lang == "pl":
                    em = EMOTION_LABELS_PL.get(em, em)
                row = self.batch_table.rowCount()
                self.batch_table.insertRow(row)
                self.batch_table.setItem(row, 0, QTableWidgetItem(str(row + 1)))
                self.batch_table.setItem(row, 1, QTableWidgetItem(str(res.age) if res.age is not None else "-"))
                self.batch_table.setItem(row, 2, QTableWidgetItem(str(res.gender) if res.gender else "-"))
                self.batch_table.setItem(row, 3, QTableWidgetItem(em))
                self.batch_table.setItem(row, 4, QTableWidgetItem(filepath.name))
            all_results.append((filepath.name, results))
        self.batch_progress.hide()
        self.btn_batch_select.setEnabled(True)
        self.btn_batch_export.setEnabled(bool(all_results))
        self.statusBar().showMessage(
            f"{_('completed')} — {len(files)} files, "
            f"{sum(len(r) for _, r in all_results)} faces"
        )

    def _export_batch_csv(self):
        path, _filter = QFileDialog.getSaveFileName(
            self, _("save_csv"), "batch_results.csv", "CSV (*.csv)"
        )
        if not path:
            return
        lang = get_language()
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.writer(f)
            writer.writerow(["File", "Face", "Age", "Gender", "Emotion"])
            for row in range(self.batch_table.rowCount()):
                vals = [self.batch_table.item(row, c).text()
                        if self.batch_table.item(row, c) else ""
                        for c in range(self.batch_table.columnCount())]
                writer.writerow(vals)
        self.statusBar().showMessage(f"Exported: {os.path.basename(path)}")

    def _toggle_webcam(self):
        if self.webcam_capture is not None:
            self.webcam_timer.stop()
            self.webcam_capture.release()
            self.webcam_capture = None
            self.webcam_pipeline = None
            self.btn_webcam.setText(_("start_camera"))
            self.webcam_label.setText(_("webcam_hint"))
            self.webcam_status.setText("")
            return
        self.webcam_pipeline = None
        self.webcam_capture = cv2.VideoCapture(0)
        if not self.webcam_capture.isOpened():
            QMessageBox.critical(self, _("error"), "Cannot open webcam")
            self.webcam_capture = None
            return
        self.btn_webcam.setText(_("stop_webcam"))
        self.webcam_timer.start(30)

    def _ensure_webcam_pipeline(self):
        detector_key = self.webcam_detector.currentText()
        backend = DETECTOR_BACKENDS.get(detector_key, "mediapipe")
        ag_model = self.ag_model_keys[self.wc_ag_model.currentIndex()]
        age = self.wc_age.isChecked()
        gender = self.wc_gender.isChecked()
        emotion = self.wc_emotion.isChecked()
        if self.webcam_pipeline is None:
            self.webcam_pipeline = AnalysisPipeline(
                detector_backend=backend,
                age_gender_model=ag_model,
                enable_age=age, enable_gender=gender,
                enable_emotion=emotion,
            )

    def _webcam_frame(self):
        if self.webcam_capture is None:
            return
        ret, frame = self.webcam_capture.read()
        if not ret:
            return
        self._ensure_webcam_pipeline()
        try:
            results = self.webcam_pipeline.analyze(frame)
        except Exception:
            results = []
        lang = get_language()
        annotated = draw_results(frame, results, lang)
        pixmap = numpy_to_pixmap(annotated, max_size=640)
        self.webcam_label.setPixmap(pixmap)
        self.webcam_label.setText("")
        if results:
            self.webcam_status.setText(
                f"{len(results)} face(s) detected"
            )
        else:
            self.webcam_status.setText(_("no_faces"))

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent):
        urls = event.mimeData().urls()
        if urls:
            path = urls[0].toLocalFile()
            ext = os.path.splitext(path)[1].lower()
            if ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"):
                self._load_image(path)
                self.tabs.setCurrentIndex(0)

    def closeEvent(self, event):
        if self.webcam_capture is not None:
            self.webcam_timer.stop()
            self.webcam_capture.release()
        if self.pipeline:
            self.pipeline.cleanup()
        if self.webcam_pipeline:
            self.webcam_pipeline.cleanup()
        event.accept()
