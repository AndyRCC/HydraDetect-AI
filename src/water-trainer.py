"""
Entrenador para dataset real — WaterHammer Real Dataset Trainer

Instrucciones:
- Guarda como: train_real_dataset_gui.py
- Dependencias:
    pip install pyqt5 matplotlib numpy scipy scikit-learn pywavelets pandas joblib
- Ejecutar:
    python3 train_real_dataset_gui.py

Características:
- Botón "Cargar CSV (sin bypass)" -> carga varios archivos que serán la clase 0 (no bypass)
- Botón "Cargar CSV (con bypass)" -> clase 1
- Data augmentation automático para alcanzar el tamaño de dataset deseado
- Selección de parámetros: tamaño dataset final, estimadores RF, C SVM, test_size, selector de modelo
- Entrena y muestra reporte; permite guardar modelo (.joblib)
"""

import sys, os
import numpy as np
import scipy.signal as signal
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq
import pywt
from PyQt5 import QtWidgets, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import pandas as pd

# ----------------------- Reuso de funciones de extracción (adaptadas) -----------------------
def extract_features(t, p, fs):
    """Extrae características de tiempo, espectrales y wavelet (misma lógica que en tu simulador)."""
    features = {}
    # baseline
    p0 = np.median(p[:max(1, int(0.05*len(p)))])
    features['baseline'] = float(p0)
    # peak
    peak_idx = int(np.argmax(p))
    features['peak_amp'] = float(p[peak_idx] - p0)
    features['t_peak'] = float(t[peak_idx]) if len(t)>0 else 0.0
    # decay tau (intento de ajuste exponencial)
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
    # spectral
    N = len(p)
    if N <= 1:
        yf = np.array([0.0])
        xf = np.array([0.0])
    else:
        yf = np.abs(rfft(p - p0))
        xf = rfftfreq(N, 1.0/fs)
    total_energy = float(np.sum(yf**2))
    features['energy_total'] = total_energy
    bands = [(0,20),(20,100),(100,500),(500,1000)]
    for (a,b) in bands:
        if len(xf) > 0:
            idxb = (xf >= a) & (xf < b)
            features[f'energy_band_{a}_{b}'] = float(np.sum(yf[idxb]**2))
        else:
            features[f'energy_band_{a}_{b}'] = 0.0
    dom = float(xf[np.argmax(yf)]) if len(xf) > 0 else 0.0
    features['dom_freq'] = dom
    # wavelet energies
    try:
        coeffs = pywt.wavedec(p - p0, 'db4', level=4)
        for i, c in enumerate(coeffs):
            features[f'wavelet_E_{i}'] = float(np.sum(np.array(c)**2))
    except Exception:
        for i in range(5):
            features[f'wavelet_E_{i}'] = 0.0
    return features

# ----------------------- Data augmentation helpers -----------------------
def add_noise(p, noise_std):
    return p + np.random.normal(0, noise_std, size=p.shape)

def scale_amplitude(p, factor):
    return p * factor

def time_shift(t, p, shift_s):
    # shift the signal in time by roll and fill edges by baseline
    dt = np.mean(np.diff(t)) if len(t)>1 else 0.0
    if dt == 0:
        return t, p
    shift_samples = int(round(shift_s / dt))
    p2 = np.roll(p, shift_samples)
    if shift_samples > 0:
        p2[:shift_samples] = p[0]
    elif shift_samples < 0:
        p2[shift_samples:] = p[-1]
    return t.copy(), p2

def time_stretch(t, p, stretch_factor):
    # simple resample-based time-stretch (nearest approach)
    if stretch_factor == 1.0 or len(t) < 2:
        return t.copy(), p.copy()
    N = len(p)
    new_len = max(2, int(N * stretch_factor))
    new_t = np.linspace(t[0], t[-1], new_len)
    new_p = np.interp(new_t, t, p)
    # resample back to original length to keep consistent feature extraction length if needed
    res_t = np.linspace(new_t[0], new_t[-1], N)
    res_p = np.interp(res_t, new_t, new_p)
    return res_t, res_p

def augment_single_signal(t, p, fs, n_aug=5, noise_range=(0.002,0.02), amp_range=(0.9,1.12),
                          shift_seconds=(-0.02, 0.02), stretch_range=(0.95,1.05)):
    """Genera n_aug versiones augmentadas de una señal real (incluye la original al inicio)."""
    augmented = []
    # include original
    augmented.append((t.copy(), p.copy()))
    for i in range(n_aug):
        p1 = p.copy()
        t1 = t.copy()
        # aplicar aleatoriamente transformaciones
        if np.random.rand() < 0.9:
            ns = np.random.uniform(noise_range[0], noise_range[1])
            p1 = add_noise(p1, ns)
        if np.random.rand() < 0.7:
            factor = np.random.uniform(amp_range[0], amp_range[1])
            p1 = scale_amplitude(p1, factor)
        if np.random.rand() < 0.6:
            shift = np.random.uniform(shift_seconds[0], shift_seconds[1])
            t1, p1 = time_shift(t1, p1, shift)
        if np.random.rand() < 0.5:
            stretch = np.random.uniform(stretch_range[0], stretch_range[1])
            t1, p1 = time_stretch(t1, p1, stretch)
        augmented.append((t1, p1))
    return augmented

# ----------------------- GUI -----------------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=3, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = fig.add_subplot(111)
        super().__init__(fig)
        fig.tight_layout()

class RealDatasetTrainer(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Entrenador - Dataset Real (WaterHammer)")
        self.setMinimumSize(1000,700)

        # datos cargados: listas de (path, t_array, p_array, fs)
        self.data_no = []
        self.data_yes = []

        # modelos y escalador
        self.clf = None
        self.scaler = None
        self.feature_names = None

        # Layouts
        main = QtWidgets.QWidget()
        self.setCentralWidget(main)
        h = QtWidgets.QHBoxLayout()
        main.setLayout(h)

        # Left control panel
        left = QtWidgets.QFrame()
        left.setMinimumWidth(360)
        vleft = QtWidgets.QVBoxLayout()
        left.setLayout(vleft)
        h.addWidget(left)

        # Buttons to load datasets
        btn_load_no = QtWidgets.QPushButton("Cargar CSV (sin bypass) - clase 0")
        btn_load_yes = QtWidgets.QPushButton("Cargar CSV (con bypass) - clase 1")
        vleft.addWidget(btn_load_no)
        vleft.addWidget(btn_load_yes)

        self.lbl_counts = QtWidgets.QLabel("Archivos cargados: clase0=0  clase1=0")
        vleft.addWidget(self.lbl_counts)

        # preview listboxes
        vleft.addWidget(QtWidgets.QLabel("Previsualización archivos (clase0):"))
        self.list_no = QtWidgets.QListWidget()
        vleft.addWidget(self.list_no, stretch=1)
        vleft.addWidget(QtWidgets.QLabel("Previsualización archivos (clase1):"))
        self.list_yes = QtWidgets.QListWidget()
        vleft.addWidget(self.list_yes, stretch=1)

        # Augmentation & training params
        vleft.addWidget(QtWidgets.QLabel("<b>Parámetros de aumentación y entrenamiento</b>"))
        form = QtWidgets.QFormLayout()
        self.target_size_spin = QtWidgets.QSpinBox(); self.target_size_spin.setRange(10,20000); self.target_size_spin.setValue(1000)
        self.n_aug_spin = QtWidgets.QSpinBox(); self.n_aug_spin.setRange(0,50); self.n_aug_spin.setValue(5)
        self.n_estimators_spin = QtWidgets.QSpinBox(); self.n_estimators_spin.setRange(10,1000); self.n_estimators_spin.setValue(100)
        self.svm_c_spin = QtWidgets.QDoubleSpinBox(); self.svm_c_spin.setRange(0.01,100.0); self.svm_c_spin.setValue(1.0)
        self.test_size_spin = QtWidgets.QDoubleSpinBox(); self.test_size_spin.setRange(0.05,0.5); self.test_size_spin.setSingleStep(0.05); self.test_size_spin.setValue(0.25)
        self.model_choice = QtWidgets.QComboBox(); self.model_choice.addItems(['RandomForest','SVM','Ambos'])
        form.addRow("Tamaño dataset final (total):", self.target_size_spin)
        form.addRow("Augmentaciones por señal (n_aug):", self.n_aug_spin)
        form.addRow("Modelo:", self.model_choice)
        form.addRow("N estimadores RF:", self.n_estimators_spin)
        form.addRow("C para SVM:", self.svm_c_spin)
        form.addRow("Test size (fracción):", self.test_size_spin)
        vleft.addLayout(form)

        btn_train = QtWidgets.QPushButton("Aumentar y Entrenar")
        btn_save_model = QtWidgets.QPushButton("Guardar modelo (.joblib)")
        vleft.addWidget(btn_train)
        vleft.addWidget(btn_save_model)
        vleft.addStretch()
        self.status = QtWidgets.QStatusBar()
        vleft.addWidget(self.status)

        # Center: plots & preview
        center = QtWidgets.QFrame()
        vcenter = QtWidgets.QVBoxLayout()
        center.setLayout(vcenter)
        h.addWidget(center, stretch=1)

        self.canvas = MplCanvas(self, width=6, height=3)
        vcenter.addWidget(self.canvas)
        # signal preview controls
        ctrl_preview = QtWidgets.QHBoxLayout()
        self.btn_preview_no = QtWidgets.QPushButton("Preview seleccionado clase0")
        self.btn_preview_yes = QtWidgets.QPushButton("Preview seleccionado clase1")
        ctrl_preview.addWidget(self.btn_preview_no); ctrl_preview.addWidget(self.btn_preview_yes)
        vcenter.addLayout(ctrl_preview)

        # Right: report
        right = QtWidgets.QFrame()
        right.setMinimumWidth(360)
        vright = QtWidgets.QVBoxLayout()
        right.setLayout(vright)
        h.addWidget(right)

        vright.addWidget(QtWidgets.QLabel("<b>Reporte de entrenamiento</b>"))
        self.report_text = QtWidgets.QPlainTextEdit()
        self.report_text.setReadOnly(True)
        vright.addWidget(self.report_text, stretch=1)

        vright.addWidget(QtWidgets.QLabel("<b>Detalle dataset cargado</b>"))
        self.dataset_info = QtWidgets.QPlainTextEdit()
        self.dataset_info.setReadOnly(True)
        vright.addWidget(self.dataset_info, stretch=1)

        # Connections
        btn_load_no.clicked.connect(self.load_files_no)
        btn_load_yes.clicked.connect(self.load_files_yes)
        btn_train.clicked.connect(self.on_augment_and_train)
        self.btn_preview_no.clicked.connect(self.preview_selected_no)
        self.btn_preview_yes.clicked.connect(self.preview_selected_yes)
        btn_save_model.clicked.connect(self.on_save_model)

    # ----------------------- IO y parsing -----------------------
    def load_csv_signal(self, path):
        """Intenta leer CSV y devolver (t_array, p_array). Si hay más columnas, intenta detectar 't' y 'p'."""
        try:
            df = pd.read_csv(path)
        except Exception:
            # fallback: try numpy
            try:
                data = np.loadtxt(path, delimiter=',')
                if data.ndim == 1 or data.shape[1] < 2:
                    raise ValueError("CSV no tiene 2 columnas.")
                t = data[:,0]; p = data[:,1]
                return t, p
            except Exception as e:
                raise ValueError(f"No se pudo leer {path}: {e}")

        # detectar columnas
        cols = [c.lower() for c in df.columns]
        if 't' in cols and 'p' in cols:
            t = df.iloc[:, cols.index('t')].to_numpy(dtype=float)
            p = df.iloc[:, cols.index('p')].to_numpy(dtype=float)
            return t, p
        # buscar nombres comunes
        possible_t = [c for c in cols if 'time' in c or 't(' in c or 'tiempo' in c]
        possible_p = [c for c in cols if 'pres' in c or 'p(' in c or 'pressure' in c or 'presion' in c]
        if possible_t and possible_p:
            t = df[possible_t[0]].to_numpy(dtype=float)
            p = df[possible_p[0]].to_numpy(dtype=float)
            return t, p
        # última opción: tomar las dos primeras columnas
        if df.shape[1] >= 2:
            t = df.iloc[:,0].to_numpy(dtype=float)
            p = df.iloc[:,1].to_numpy(dtype=float)
            return t, p
        raise ValueError("No se pudieron identificar columnas 't' y 'p' en el CSV.")

    def infer_fs(self, t):
        if len(t) < 2:
            return 1000
        dt = np.mean(np.diff(t))
        if dt <= 0:
            return 1000
        return int(round(1.0/dt))

    def load_files_no(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Seleccionar CSV (sin bypass)", "", "CSV files (*.csv);;All files (*)")
        if not files:
            return
        added = 0
        for f in files:
            try:
                t, p = self.load_csv_signal(f)
                fs = self.infer_fs(t)
                self.data_no.append((f, t, p, fs))
                self.list_no.addItem(os.path.basename(f))
                added += 1
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error al leer", f"No se pudo leer {f}:\n{str(e)}")
        self.update_counts()
        self.update_dataset_info()
        self.status.showMessage(f"Añadidos {added} archivos a clase 0")

    def load_files_yes(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(self, "Seleccionar CSV (con bypass)", "", "CSV files (*.csv);;All files (*)")
        if not files:
            return
        added = 0
        for f in files:
            try:
                t, p = self.load_csv_signal(f)
                fs = self.infer_fs(t)
                self.data_yes.append((f, t, p, fs))
                self.list_yes.addItem(os.path.basename(f))
                added += 1
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error al leer", f"No se pudo leer {f}:\n{str(e)}")
        self.update_counts()
        self.update_dataset_info()
        self.status.showMessage(f"Añadidos {added} archivos a clase 1")

    def update_counts(self):
        self.lbl_counts.setText(f"Archivos cargados: clase0={len(self.data_no)}  clase1={len(self.data_yes)}")

    def update_dataset_info(self):
        txt = f"Clase 0: {len(self.data_no)} archivos\n"
        for (p,t_arr,p_arr,fs) in self.data_no[:5]:
            txt += f"  {os.path.basename(p)}  fs={fs}  len={len(t_arr)}\n"
        txt += f"\nClase 1: {len(self.data_yes)} archivos\n"
        for (p,t_arr,p_arr,fs) in self.data_yes[:5]:
            txt += f"  {os.path.basename(p)}  fs={fs}  len={len(t_arr)}\n"
        self.dataset_info.setPlainText(txt)

    # ----------------------- Preview -----------------------
    def preview_selected_no(self):
        it = self.list_no.currentRow()
        if it < 0 or it >= len(self.data_no):
            QtWidgets.QMessageBox.information(self, "Preview", "Seleccione un archivo en la lista clase0")
            return
        _, t, p, fs = self.data_no[it]
        self.plot_signal(t, p, title=f"Preview clase0: {os.path.basename(self.data_no[it][0])}")

    def preview_selected_yes(self):
        it = self.list_yes.currentRow()
        if it < 0 or it >= len(self.data_yes):
            QtWidgets.QMessageBox.information(self, "Preview", "Seleccione un archivo en la lista clase1")
            return
        _, t, p, fs = self.data_yes[it]
        self.plot_signal(t, p, title=f"Preview clase1: {os.path.basename(self.data_yes[it][0])}")

    def plot_signal(self, t, p, title="Señal"):
        self.canvas.ax.clear()
        self.canvas.ax.plot(t, p)
        self.canvas.ax.set_title(title)
        self.canvas.ax.set_xlabel("t (s)")
        self.canvas.ax.set_ylabel("Presión")
        self.canvas.draw()

    # ----------------------- Augment + train -----------------------
    def on_augment_and_train(self):
        if len(self.data_no) == 0 or len(self.data_yes) == 0:
            QtWidgets.QMessageBox.warning(self, "Datos insuficientes", "Necesitas cargar al menos un archivo por clase (sin bypass y con bypass).")
            return
        target_total = int(self.target_size_spin.value())
        n_aug = int(self.n_aug_spin.value())
        model_choice = self.model_choice.currentText()
        n_estimators = int(self.n_estimators_spin.value())
        svm_c = float(self.svm_c_spin.value())
        test_size = float(self.test_size_spin.value())

        self.status.showMessage("Generando dataset con aumentación...")
        QtWidgets.QApplication.processEvents()

        # generar lista de (features, label)
        X_list = []
        y_list = []

        # strategy: generar augmentaciones por señal y luego balancear hasta target_total
        def process_group(data_group, label):
            feats_list = []
            for (path, t, p, fs) in data_group:
                augs = augment_single_signal(t, p, fs, n_aug=n_aug)
                for (t2, p2) in augs:
                    try:
                        fdict = extract_features(t2, p2, fs)
                        feats_list.append(fdict)
                    except Exception:
                        continue
            return feats_list

        feats_no = process_group(self.data_no, 0)
        feats_yes = process_group(self.data_yes, 1)

        if len(feats_no) == 0 or len(feats_yes) == 0:
            QtWidgets.QMessageBox.warning(self, "Error extracción", "No se pudieron extraer características de las señales cargadas.")
            return

        # Default feature order
        feature_names = sorted(feats_no[0].keys())
        self.feature_names = feature_names

        # convertir dicts a arrays
        X_no = np.array([[f[n] for n in feature_names] for f in feats_no])
        X_yes = np.array([[f[n] for n in feature_names] for f in feats_yes])
        n_no = X_no.shape[0]
        n_yes = X_yes.shape[0]

        # balancear y muestrear para alcanzar target_total (mantener balance 50/50)
        half = target_total // 2
        def sample_or_repeat(Xgroup, N):
            if Xgroup.shape[0] >= N:
                idx = np.random.choice(Xgroup.shape[0], size=N, replace=False)
                return Xgroup[idx]
            else:
                # repetir con reemplazo
                idx = np.random.choice(Xgroup.shape[0], size=N, replace=True)
                return Xgroup[idx]

        X_no_final = sample_or_repeat(X_no, half)
        X_yes_final = sample_or_repeat(X_yes, target_total - half)
        X = np.vstack([X_no_final, X_yes_final])
        y = np.hstack([np.zeros(len(X_no_final), dtype=int), np.ones(len(X_yes_final), dtype=int)])

        # mezclar
        perm = np.random.permutation(len(y))
        X = X[perm]
        y = y[perm]

        # Escalado
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        self.scaler = scaler

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=test_size, random_state=42, stratify=y)

        report_text = ""
        # Entrenamiento RF
        if model_choice in ('RandomForest','Ambos'):
            rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
            rf.fit(X_train, y_train)
            preds = rf.predict(X_test)
            acc = accuracy_score(y_test, preds)
            report_text += f"RandomForest (n_estimators={n_estimators})\nAccuracy: {acc:.4f}\n"
            report_text += classification_report(y_test, preds)
            # guardar si solo RF
            if model_choice == 'RandomForest':
                self.clf = {'rf': rf}
            else:
                if self.clf is None:
                    self.clf = {}
                self.clf['rf'] = rf

        # Entrenamiento SVM
        if model_choice in ('SVM','Ambos'):
            svm = SVC(kernel='rbf', probability=True, C=svm_c)
            svm.fit(X_train, y_train)
            preds = svm.predict(X_test)
            acc = accuracy_score(y_test, preds)
            report_text += f"\nSVM (C={svm_c})\nAccuracy: {acc:.4f}\n"
            report_text += classification_report(y_test, preds)
            if model_choice == 'SVM':
                self.clf = {'svm': svm}
            else:
                if self.clf is None:
                    self.clf = {}
                self.clf['svm'] = svm

        # guardar feature_names y scaler y feature order
        self.feature_names = feature_names
        self.report_text.setPlainText(report_text)
        self.status.showMessage("Entrenamiento finalizado.")
        # Mostrar info resumida del dataset
        info = f"Dataset final: {len(y)} muestras (clase0={len(X_no_final)} clase1={len(X_yes_final)})\n"
        info += f"Feature order: {', '.join(self.feature_names)}\n"
        info += f"Nota: modelos guardados internamente en memoria. Use 'Guardar modelo' para exportar .joblib\n"
        self.dataset_info.setPlainText(info)

    # ----------------------- Guardar modelo -----------------------
    def on_save_model(self):
        if self.clf is None or self.feature_names is None or self.scaler is None:
            QtWidgets.QMessageBox.warning(self, "No hay modelo", "No hay modelo entrenado para guardar. Entrene primero.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Guardar modelo", "", "Joblib files (*.joblib)")
        if not path:
            return
        payload = {
            'models': self.clf,
            'scaler': self.scaler,
            'feature_names': self.feature_names
        }
        try:
            joblib.dump(payload, path)
            self.status.showMessage(f"Modelo guardado en {path}")
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Error al guardar", str(e))

# ----------------------- Main -----------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    win = RealDatasetTrainer()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
