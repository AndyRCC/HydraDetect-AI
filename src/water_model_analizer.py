"""
Evaluador de Modelos — WaterHammer Model Evaluator
====================================================
Script independiente para cargar modelos .joblib entrenados con el
"Entrenador - Dataset Real (WaterHammer)" y evaluarlos con datos CSV nuevos.

Dependencias:
    pip install pyqt5 matplotlib numpy scipy scikit-learn pywavelets pandas joblib

Ejecutar:
    python3 evaluate_waterhammer_model.py
"""

import sys, os
import numpy as np
import scipy.signal as signal
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq
import pywt
from PyQt5 import QtWidgets, QtCore, QtGui
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib import gridspec
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    classification_report, accuracy_score,
    roc_curve, auc, RocCurveDisplay
)
import joblib
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# Extracción de features (idéntica al entrenador)
# ══════════════════════════════════════════════════════════════════════════════
def extract_features(t, p, fs):
    features = {}
    p0 = np.median(p[:max(1, int(0.05 * len(p)))])
    features['baseline'] = float(p0)
    peak_idx = int(np.argmax(p))
    features['peak_amp'] = float(p[peak_idx] - p0)
    features['t_peak'] = float(t[peak_idx]) if len(t) > 0 else 0.0
    try:
        idx_fit = (t >= t[peak_idx]) & (t <= t[peak_idx] + 1.0)
        if np.sum(idx_fit) > 10:
            env = np.abs(p[idx_fit] - p0)
            env[env <= 1e-6] = 1e-6
            def expo(x, a, tau):
                return a * np.exp(-(x - t[peak_idx]) / tau)
            popt, _ = curve_fit(expo, t[idx_fit], env, p0=[env[0], 0.3], maxfev=5000)
            features['decay_tau'] = float(max(popt[1], 1e-3))
        else:
            features['decay_tau'] = 0.0
    except Exception:
        features['decay_tau'] = 0.0
    N = len(p)
    if N <= 1:
        yf = np.array([0.0]); xf = np.array([0.0])
    else:
        yf = np.abs(rfft(p - p0)); xf = rfftfreq(N, 1.0 / fs)
    total_energy = float(np.sum(yf ** 2))
    features['energy_total'] = total_energy
    bands = [(0, 20), (20, 100), (100, 500), (500, 1000)]
    for (a, b) in bands:
        if len(xf) > 0:
            idxb = (xf >= a) & (xf < b)
            features[f'energy_band_{a}_{b}'] = float(np.sum(yf[idxb] ** 2))
        else:
            features[f'energy_band_{a}_{b}'] = 0.0
    dom = float(xf[np.argmax(yf)]) if len(xf) > 0 else 0.0
    features['dom_freq'] = dom
    try:
        coeffs = pywt.wavedec(p - p0, 'db4', level=4)
        for i, c in enumerate(coeffs):
            features[f'wavelet_E_{i}'] = float(np.sum(np.array(c) ** 2))
    except Exception:
        for i in range(5):
            features[f'wavelet_E_{i}'] = 0.0
    return features


# ══════════════════════════════════════════════════════════════════════════════
# Canvas de Matplotlib
# ══════════════════════════════════════════════════════════════════════════════
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=7, height=5, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi, facecolor='#0f1117')
        super().__init__(self.fig)
        self.setStyleSheet("background-color: #0f1117;")

    def clear_axes(self):
        self.fig.clear()
        self.draw()


# ══════════════════════════════════════════════════════════════════════════════
# Ventana principal
# ══════════════════════════════════════════════════════════════════════════════
DARK_BG      = "#0f1117"
PANEL_BG     = "#1a1d27"
ACCENT       = "#00d4aa"
ACCENT2      = "#ff6b6b"
TEXT_PRIMARY = "#e8eaf0"
TEXT_MUTED   = "#8b8fa8"
BORDER       = "#2a2d3e"

STYLESHEET = f"""
QMainWindow, QWidget {{
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
    font-family: 'Courier New', monospace;
    font-size: 12px;
}}
QFrame#panel {{
    background-color: {PANEL_BG};
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
QPushButton {{
    background-color: {PANEL_BG};
    color: {ACCENT};
    border: 1px solid {ACCENT};
    border-radius: 4px;
    padding: 7px 14px;
    font-weight: bold;
    font-size: 11px;
}}
QPushButton:hover {{
    background-color: {ACCENT};
    color: {DARK_BG};
}}
QPushButton#danger {{
    color: {ACCENT2};
    border-color: {ACCENT2};
}}
QPushButton#danger:hover {{
    background-color: {ACCENT2};
    color: {DARK_BG};
}}
QComboBox, QListWidget, QPlainTextEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {DARK_BG};
    color: {TEXT_PRIMARY};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 4px;
}}
QComboBox::drop-down {{ border: none; }}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    background-color: {PANEL_BG};
    border-radius: 6px;
}}
QTabBar::tab {{
    background-color: {DARK_BG};
    color: {TEXT_MUTED};
    padding: 8px 18px;
    border-bottom: 2px solid transparent;
    font-weight: bold;
}}
QTabBar::tab:selected {{
    color: {ACCENT};
    border-bottom: 2px solid {ACCENT};
    background-color: {PANEL_BG};
}}
QLabel#title {{
    font-size: 18px;
    font-weight: bold;
    color: {ACCENT};
    letter-spacing: 2px;
}}
QLabel#subtitle {{
    font-size: 10px;
    color: {TEXT_MUTED};
    letter-spacing: 1px;
}}
QLabel#section {{
    font-size: 11px;
    font-weight: bold;
    color: {ACCENT};
    padding: 4px 0px;
}}
QStatusBar {{
    background-color: {PANEL_BG};
    color: {TEXT_MUTED};
    border-top: 1px solid {BORDER};
    font-size: 10px;
}}
QSplitter::handle {{
    background-color: {BORDER};
}}
QScrollBar:vertical {{
    background: {DARK_BG};
    width: 8px;
}}
QScrollBar::handle:vertical {{
    background: {BORDER};
    border-radius: 4px;
}}
"""


def make_panel():
    f = QtWidgets.QFrame()
    f.setObjectName("panel")
    return f


class EvaluatorWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("WaterHammer — Evaluador de Modelos")
        self.setMinimumSize(1280, 780)
        self.setStyleSheet(STYLESHEET)

        # Estado interno
        self.payload   = None   # dict cargado desde .joblib
        self.data_no   = []     # list of (path, t, p, fs)
        self.data_yes  = []
        self.last_eval = {}     # resultados de la última evaluación

        self._build_ui()

    # ─────────────────────────────────────────────────────────────
    # Construcción de UI
    # ─────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        vroot = QtWidgets.QVBoxLayout(root)
        vroot.setContentsMargins(12, 12, 12, 6)
        vroot.setSpacing(8)

        # ── Header ──────────────────────────────────────────────
        hdr = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("WATERHAMMER  MODEL  EVALUATOR")
        title.setObjectName("title")
        sub   = QtWidgets.QLabel("Carga un .joblib + CSVs de prueba y evalúa RF y/o SVM")
        sub.setObjectName("subtitle")
        hdr.addWidget(title)
        hdr.addStretch()
        hdr.addWidget(sub)
        vroot.addLayout(hdr)

        # Separador decorativo
        sep = QtWidgets.QFrame()
        sep.setFrameShape(QtWidgets.QFrame.HLine)
        sep.setStyleSheet(f"color: {BORDER};")
        vroot.addWidget(sep)

        # ── Splitter principal ───────────────────────────────────
        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        vroot.addWidget(splitter, stretch=1)

        # Panel izquierdo (controles)
        left = make_panel()
        left.setMaximumWidth(320)
        left.setMinimumWidth(280)
        vleft = QtWidgets.QVBoxLayout(left)
        vleft.setContentsMargins(12, 12, 12, 12)
        vleft.setSpacing(8)
        splitter.addWidget(left)

        # Panel central (plots tabs)
        center = make_panel()
        vcenter = QtWidgets.QVBoxLayout(center)
        vcenter.setContentsMargins(8, 8, 8, 8)
        self.tabs = QtWidgets.QTabWidget()
        vcenter.addWidget(self.tabs)
        splitter.addWidget(center)
        splitter.setStretchFactor(1, 2)

        # ── Panel izquierdo — contenido ─────────────────────────
        # 1. Cargar modelo
        lbl_model = QtWidgets.QLabel("MODELO"); lbl_model.setObjectName("section")
        vleft.addWidget(lbl_model)

        self.btn_load_model = QtWidgets.QPushButton("⬆  Cargar .joblib")
        vleft.addWidget(self.btn_load_model)

        self.lbl_model_info = QtWidgets.QLabel("Sin modelo cargado")
        self.lbl_model_info.setWordWrap(True)
        self.lbl_model_info.setStyleSheet(f"color:{TEXT_MUTED}; font-size:10px;")
        vleft.addWidget(self.lbl_model_info)

        vleft.addWidget(self._hline())

        # 2. Cargar CSVs
        lbl_data = QtWidgets.QLabel("DATOS DE PRUEBA"); lbl_data.setObjectName("section")
        vleft.addWidget(lbl_data)

        self.btn_load_no  = QtWidgets.QPushButton("Cargar CSV  clase 0  (sin bypass)")
        self.btn_load_yes = QtWidgets.QPushButton("Cargar CSV  clase 1  (con bypass)")
        vleft.addWidget(self.btn_load_no)
        vleft.addWidget(self.btn_load_yes)

        self.lbl_counts = QtWidgets.QLabel("clase0: 0 archivos   |   clase1: 0 archivos")
        self.lbl_counts.setStyleSheet(f"color:{TEXT_MUTED}; font-size:10px;")
        vleft.addWidget(self.lbl_counts)

        # Listas de archivos
        vleft.addWidget(QtWidgets.QLabel("Clase 0:"))
        self.list_no = QtWidgets.QListWidget(); self.list_no.setMaximumHeight(80)
        vleft.addWidget(self.list_no)
        vleft.addWidget(QtWidgets.QLabel("Clase 1:"))
        self.list_yes = QtWidgets.QListWidget(); self.list_yes.setMaximumHeight(80)
        vleft.addWidget(self.list_yes)

        # Botones preview señal
        row_prev = QtWidgets.QHBoxLayout()
        self.btn_prev_no  = QtWidgets.QPushButton("Ver clase 0")
        self.btn_prev_yes = QtWidgets.QPushButton("Ver clase 1")
        row_prev.addWidget(self.btn_prev_no); row_prev.addWidget(self.btn_prev_yes)
        vleft.addLayout(row_prev)

        vleft.addWidget(self._hline())

        # 3. Selección de modelo a evaluar
        lbl_sel = QtWidgets.QLabel("EVALUAR MODELO"); lbl_sel.setObjectName("section")
        vleft.addWidget(lbl_sel)
        self.combo_model = QtWidgets.QComboBox()
        self.combo_model.addItem("(sin modelo cargado)")
        vleft.addWidget(self.combo_model)

        self.btn_evaluate = QtWidgets.QPushButton("▶  EVALUAR")
        self.btn_evaluate.setMinimumHeight(36)
        vleft.addWidget(self.btn_evaluate)

        # Limpiar todo
        self.btn_clear = QtWidgets.QPushButton("✕  Limpiar todo")
        self.btn_clear.setObjectName("danger")
        vleft.addWidget(self.btn_clear)

        vleft.addStretch()

        # ── Tabs de resultados ───────────────────────────────────
        self._setup_tabs()

        # ── Status bar ──────────────────────────────────────────
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)
        self.status.showMessage("Listo. Carga un modelo .joblib para comenzar.")

        # ── Conexiones ───────────────────────────────────────────
        self.btn_load_model.clicked.connect(self.on_load_model)
        self.btn_load_no.clicked.connect(self.on_load_no)
        self.btn_load_yes.clicked.connect(self.on_load_yes)
        self.btn_prev_no.clicked.connect(self.on_preview_no)
        self.btn_prev_yes.clicked.connect(self.on_preview_yes)
        self.btn_evaluate.clicked.connect(self.on_evaluate)
        self.btn_clear.clicked.connect(self.on_clear)

    def _setup_tabs(self):
        # Tab: Preview de señal
        tab_sig = QtWidgets.QWidget()
        vt = QtWidgets.QVBoxLayout(tab_sig)
        self.canvas_signal = MplCanvas(width=7, height=4)
        vt.addWidget(self.canvas_signal)
        self.tabs.addTab(tab_sig, "📈 Señal")

        # Tab: Matriz de confusión
        tab_cm = QtWidgets.QWidget()
        vt2 = QtWidgets.QVBoxLayout(tab_cm)
        self.canvas_cm = MplCanvas(width=7, height=5)
        vt2.addWidget(self.canvas_cm)
        self.tabs.addTab(tab_cm, "🔲 Matriz Confusión")

        # Tab: Curva ROC
        tab_roc = QtWidgets.QWidget()
        vt3 = QtWidgets.QVBoxLayout(tab_roc)
        self.canvas_roc = MplCanvas(width=7, height=5)
        vt3.addWidget(self.canvas_roc)
        self.tabs.addTab(tab_roc, "📉 Curva ROC")

        # Tab: Feature Importance (RF)
        tab_fi = QtWidgets.QWidget()
        vt4 = QtWidgets.QVBoxLayout(tab_fi)
        self.canvas_fi = MplCanvas(width=7, height=5)
        vt4.addWidget(self.canvas_fi)
        self.tabs.addTab(tab_fi, "📊 Feature Importance")

        # Tab: Reporte de clasificación
        tab_rep = QtWidgets.QWidget()
        vt5 = QtWidgets.QVBoxLayout(tab_rep)
        self.report_text = QtWidgets.QPlainTextEdit()
        self.report_text.setReadOnly(True)
        self.report_text.setStyleSheet(
            f"background:{DARK_BG}; color:{ACCENT}; font-family:'Courier New'; font-size:11px;"
        )
        vt5.addWidget(self.report_text)
        self.tabs.addTab(tab_rep, "📋 Reporte")

    def _hline(self):
        f = QtWidgets.QFrame()
        f.setFrameShape(QtWidgets.QFrame.HLine)
        f.setStyleSheet(f"color:{BORDER};")
        return f

    # ─────────────────────────────────────────────────────────────
    # IO helpers
    # ─────────────────────────────────────────────────────────────
    def _load_csv(self, path):
        try:
            df = pd.read_csv(path)
        except Exception:
            data = np.loadtxt(path, delimiter=',')
            return data[:, 0], data[:, 1]
        cols = [c.lower() for c in df.columns]
        if 't' in cols and 'p' in cols:
            return df.iloc[:, cols.index('t')].to_numpy(float), df.iloc[:, cols.index('p')].to_numpy(float)
        pt = [c for c in cols if 'time' in c or 'tiempo' in c]
        pp = [c for c in cols if 'pres' in c or 'pressure' in c or 'presion' in c]
        if pt and pp:
            return df[pt[0]].to_numpy(float), df[pp[0]].to_numpy(float)
        return df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float)

    def _infer_fs(self, t):
        if len(t) < 2: return 1000
        dt = np.mean(np.diff(t))
        return int(round(1.0 / dt)) if dt > 0 else 1000

    # ─────────────────────────────────────────────────────────────
    # Slots
    # ─────────────────────────────────────────────────────────────
    def on_load_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Cargar modelo .joblib", "", "Joblib files (*.joblib);;All (*)"
        )
        if not path:
            return
        try:
            self.payload = joblib.load(path)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"No se pudo cargar el modelo:\n{e}")
            return

        # Validar estructura
        if not all(k in self.payload for k in ('models', 'scaler', 'feature_names')):
            QtWidgets.QMessageBox.critical(
                self, "Formato inválido",
                "El .joblib no tiene la estructura esperada:\n{'models', 'scaler', 'feature_names'}"
            )
            self.payload = None
            return

        models_found = list(self.payload['models'].keys())
        n_features   = len(self.payload['feature_names'])

        self.lbl_model_info.setText(
            f"✅ {os.path.basename(path)}\n"
            f"Modelos: {', '.join(models_found).upper()}\n"
            f"Features: {n_features}"
        )
        self.lbl_model_info.setStyleSheet(f"color:{ACCENT}; font-size:10px;")

        # Poblar combo
        self.combo_model.clear()
        label_map = {'rf': 'Random Forest', 'svm': 'SVM'}
        for k in models_found:
            self.combo_model.addItem(label_map.get(k, k.upper()), userData=k)
        if len(models_found) > 1:
            self.combo_model.addItem("Ambos", userData="both")

        self.status.showMessage(f"Modelo cargado: {os.path.basename(path)}  |  Modelos disponibles: {models_found}")

    def on_load_no(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "CSV clase 0 (sin bypass)", "", "CSV (*.csv);;All (*)"
        )
        for f in files:
            try:
                t, p = self._load_csv(f)
                fs = self._infer_fs(t)
                self.data_no.append((f, t, p, fs))
                self.list_no.addItem(os.path.basename(f))
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"{f}:\n{e}")
        self._update_counts()

    def on_load_yes(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "CSV clase 1 (con bypass)", "", "CSV (*.csv);;All (*)"
        )
        for f in files:
            try:
                t, p = self._load_csv(f)
                fs = self._infer_fs(t)
                self.data_yes.append((f, t, p, fs))
                self.list_yes.addItem(os.path.basename(f))
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error", f"{f}:\n{e}")
        self._update_counts()

    def _update_counts(self):
        self.lbl_counts.setText(
            f"clase0: {len(self.data_no)} archivos   |   clase1: {len(self.data_yes)} archivos"
        )

    def on_preview_no(self):
        idx = self.list_no.currentRow()
        if idx < 0 or idx >= len(self.data_no):
            QtWidgets.QMessageBox.information(self, "Info", "Selecciona un archivo de la lista clase0.")
            return
        _, t, p, _ = self.data_no[idx]
        self._plot_signal(t, p, f"Clase 0 — {os.path.basename(self.data_no[idx][0])}", ACCENT)
        self.tabs.setCurrentIndex(0)

    def on_preview_yes(self):
        idx = self.list_yes.currentRow()
        if idx < 0 or idx >= len(self.data_yes):
            QtWidgets.QMessageBox.information(self, "Info", "Selecciona un archivo de la lista clase1.")
            return
        _, t, p, _ = self.data_yes[idx]
        self._plot_signal(t, p, f"Clase 1 — {os.path.basename(self.data_yes[idx][0])}", ACCENT2)
        self.tabs.setCurrentIndex(0)

    def _plot_signal(self, t, p, title, color):
        fig = self.canvas_signal.fig
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor(DARK_BG)
        fig.patch.set_facecolor(DARK_BG)
        ax.plot(t, p, color=color, linewidth=1.2)
        ax.set_title(title, color=TEXT_PRIMARY, fontsize=11, pad=10)
        ax.set_xlabel("t (s)", color=TEXT_MUTED)
        ax.set_ylabel("Presión", color=TEXT_MUTED)
        ax.tick_params(colors=TEXT_MUTED)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        fig.tight_layout()
        self.canvas_signal.draw()

    def on_clear(self):
        self.payload = None
        self.data_no.clear(); self.data_yes.clear()
        self.last_eval = {}
        self.list_no.clear(); self.list_yes.clear()
        self.combo_model.clear(); self.combo_model.addItem("(sin modelo cargado)")
        self.lbl_model_info.setText("Sin modelo cargado")
        self.lbl_model_info.setStyleSheet(f"color:{TEXT_MUTED}; font-size:10px;")
        self.lbl_counts.setText("clase0: 0 archivos   |   clase1: 0 archivos")
        self.report_text.clear()
        for c in [self.canvas_signal, self.canvas_cm, self.canvas_roc, self.canvas_fi]:
            c.fig.clear(); c.draw()
        self.status.showMessage("Limpiado. Carga un modelo para comenzar.")

    # ─────────────────────────────────────────────────────────────
    # Evaluación principal
    # ─────────────────────────────────────────────────────────────
    def on_evaluate(self):
        if self.payload is None:
            QtWidgets.QMessageBox.warning(self, "Sin modelo", "Carga un .joblib primero.")
            return
        if len(self.data_no) == 0 or len(self.data_yes) == 0:
            QtWidgets.QMessageBox.warning(
                self, "Sin datos",
                "Carga al menos un CSV de cada clase (sin bypass y con bypass)."
            )
            return

        self.status.showMessage("Extrayendo características…")
        QtWidgets.QApplication.processEvents()

        scaler        = self.payload['scaler']
        feature_names = self.payload['feature_names']
        models        = self.payload['models']

        selected_key = self.combo_model.currentData()

        # ── Extracción de features ─────────────────────────────
        def extract_group(data_group, label):
            rows, labels = [], []
            for (path, t, p, fs) in data_group:
                try:
                    fdict = extract_features(t, p, fs)
                    row   = [fdict.get(n, 0.0) for n in feature_names]
                    rows.append(row); labels.append(label)
                except Exception:
                    continue
            return rows, labels

        rows0, lbl0 = extract_group(self.data_no,  0)
        rows1, lbl1 = extract_group(self.data_yes, 1)

        if len(rows0) == 0 or len(rows1) == 0:
            QtWidgets.QMessageBox.warning(self, "Error", "No se extrajeron features válidas de las señales.")
            return

        X_raw = np.array(rows0 + rows1)
        y     = np.array(lbl0  + lbl1)
        X     = scaler.transform(X_raw)

        # ── Determinar qué modelos evaluar ────────────────────
        if selected_key == "both":
            keys_to_eval = [k for k in models]
        else:
            keys_to_eval = [selected_key]

        full_report = ""
        self.last_eval = {}

        for key in keys_to_eval:
            clf   = models[key]
            preds = clf.predict(X)
            acc   = accuracy_score(y, preds)
            cm    = confusion_matrix(y, preds)
            label_name = "Random Forest" if key == "rf" else "SVM"

            has_proba = hasattr(clf, "predict_proba")
            proba = clf.predict_proba(X)[:, 1] if has_proba else None

            self.last_eval[key] = {
                'clf': clf, 'preds': preds, 'y': y,
                'cm': cm, 'proba': proba, 'acc': acc,
                'label': label_name
            }

            full_report += f"{'═'*50}\n"
            full_report += f"  {label_name}   —   Accuracy: {acc:.4f}\n"
            full_report += f"{'═'*50}\n"
            full_report += f"Muestras evaluadas: {len(y)}  (clase0={sum(y==0)}, clase1={sum(y==1)})\n\n"
            full_report += classification_report(
                y, preds, target_names=["Sin bypass (0)", "Con bypass (1)"]
            )
            full_report += "\nMatriz de Confusión:\n"
            full_report += f"  TN={cm[0,0]}  FP={cm[0,1]}\n"
            full_report += f"  FN={cm[1,0]}  TP={cm[1,1]}\n\n"

        self.report_text.setPlainText(full_report)

        # ── Dibujar plots ──────────────────────────────────────
        self._draw_confusion_matrices()
        self._draw_roc_curves()
        self._draw_feature_importance()

        self.tabs.setCurrentIndex(1)
        self.status.showMessage(f"Evaluación completada. {len(y)} muestras evaluadas.")

    # ─────────────────────────────────────────────────────────────
    # Plots
    # ─────────────────────────────────────────────────────────────
    def _apply_dark_ax(self, ax, title=""):
        ax.set_facecolor(PANEL_BG)
        if title:
            ax.set_title(title, color=TEXT_PRIMARY, fontsize=10, pad=8, fontweight='bold')
        ax.tick_params(colors=TEXT_MUTED, labelsize=9)
        for sp in ax.spines.values():
            sp.set_edgecolor(BORDER)
        ax.xaxis.label.set_color(TEXT_MUTED)
        ax.yaxis.label.set_color(TEXT_MUTED)

    def _draw_confusion_matrices(self):
        fig = self.canvas_cm.fig
        fig.clear()
        fig.patch.set_facecolor(DARK_BG)
        keys = list(self.last_eval.keys())
        n    = len(keys)
        axes = fig.subplots(1, n) if n > 1 else [fig.add_subplot(111)]

        colors_cm = ['#00d4aa', '#ff6b6b']
        import matplotlib.colors as mcolors

        for ax, key in zip(axes, keys):
            ev   = self.last_eval[key]
            cm   = ev['cm']
            acc  = ev['acc']
            cmap = mcolors.LinearSegmentedColormap.from_list(
                'dark_teal', [PANEL_BG, ACCENT], N=256
            )
            disp = ConfusionMatrixDisplay(
                confusion_matrix=cm,
                display_labels=["Sin bypass", "Con bypass"]
            )
            disp.plot(ax=ax, colorbar=False, cmap=cmap)
            self._apply_dark_ax(ax, f"{ev['label']}  (acc={acc:.3f})")
            # recolor text inside cells
            for text_obj in disp.text_.ravel():
                text_obj.set_color(TEXT_PRIMARY)
                text_obj.set_fontsize(14)
                text_obj.set_fontweight('bold')
            ax.tick_params(axis='x', rotation=20)

        fig.tight_layout(pad=2.0)
        self.canvas_cm.draw()

    def _draw_roc_curves(self):
        fig = self.canvas_roc.fig
        fig.clear()
        fig.patch.set_facecolor(DARK_BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(PANEL_BG)

        colors_roc = [ACCENT, ACCENT2, '#ffd166', '#a29bfe']
        ax.plot([0, 1], [0, 1], color=BORDER, linestyle='--', linewidth=1.2, label='Aleatorio')

        any_roc = False
        for i, (key, ev) in enumerate(self.last_eval.items()):
            if ev['proba'] is None:
                continue
            fpr, tpr, _ = roc_curve(ev['y'], ev['proba'])
            roc_auc      = auc(fpr, tpr)
            color        = colors_roc[i % len(colors_roc)]
            ax.plot(fpr, tpr, color=color, linewidth=2.0,
                    label=f"{ev['label']}  (AUC = {roc_auc:.3f})")
            # fill under curve
            ax.fill_between(fpr, tpr, alpha=0.08, color=color)
            any_roc = True

        if not any_roc:
            ax.text(0.5, 0.5, "ROC no disponible\n(SVM sin probability=True)",
                    ha='center', va='center', color=TEXT_MUTED, fontsize=11,
                    transform=ax.transAxes)

        self._apply_dark_ax(ax, "Curva ROC")
        ax.set_xlabel("Tasa de Falsos Positivos (FPR)")
        ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)")
        leg = ax.legend(facecolor=PANEL_BG, edgecolor=BORDER, labelcolor=TEXT_PRIMARY, fontsize=9)
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        fig.tight_layout()
        self.canvas_roc.draw()

    def _draw_feature_importance(self):
        fig = self.canvas_fi.fig
        fig.clear()
        fig.patch.set_facecolor(DARK_BG)
        ax = fig.add_subplot(111)
        ax.set_facecolor(PANEL_BG)

        rf_ev = self.last_eval.get('rf')
        if rf_ev is None:
            ax.text(0.5, 0.5,
                    "Feature Importance\nsolo disponible para Random Forest",
                    ha='center', va='center', color=TEXT_MUTED, fontsize=11,
                    transform=ax.transAxes)
            self._apply_dark_ax(ax, "Feature Importance (RF)")
            fig.tight_layout()
            self.canvas_fi.draw()
            return

        clf           = rf_ev['clf']
        feature_names = self.payload['feature_names']
        importances   = clf.feature_importances_
        sorted_idx    = np.argsort(importances)[-15:]  # top 15

        colors_grad = [
            f"#{int(0x00 + (0xff-0x00)*i/(len(sorted_idx)-1)):02x}"
            f"{int(0xd4 + (0x6b-0xd4)*i/(len(sorted_idx)-1)):02x}"
            f"{int(0xaa + (0x6b-0xaa)*i/(len(sorted_idx)-1)):02x}"
            for i in range(len(sorted_idx))
        ]

        bars = ax.barh(
            [feature_names[i] for i in sorted_idx],
            importances[sorted_idx],
            color=colors_grad, edgecolor='none', height=0.6
        )
        ax.set_xlabel("Importancia")
        self._apply_dark_ax(ax, "Top 15 Features — Random Forest")
        ax.tick_params(axis='y', labelsize=8)

        # Etiquetas de valor
        for bar, val in zip(bars, importances[sorted_idx]):
            ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f}", va='center', color=TEXT_MUTED, fontsize=7)

        fig.tight_layout()
        self.canvas_fi.draw()


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════
def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    win = EvaluatorWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()