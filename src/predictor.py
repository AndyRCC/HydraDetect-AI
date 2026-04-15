"""
WaterHammer GUI Simulator & Presentation Tool (PATCHED)
Compatibilidad mejorada de carga/guardado de modelos (.joblib) — acepta formatos del trainer externo

"""
import sys
import os
import io
import numpy as np
import scipy.signal as signal
from scipy.optimize import curve_fit
from scipy.fft import rfft, rfftfreq
import pywt
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5 import QtWidgets, QtCore
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
# ------------------- Signal generation and feature extraction -------------------

def generate_transient(duration=5.0, fs=2000, p0=2.5, t0=0.5, A=0.6, f0=25.0,
                       tau=0.4, noise_std=0.01, bypass=False, seed=None):
    """
    Simple phenomenological transient model: an exponentially-decaying sinusoid
    that starts at t0 and is added to the baseline pressure p0. 'bypass' modifies
    amplitude and decay to emulate effect of parallel leak/bypass.
    """
    if seed is not None:
        np.random.seed(seed)
    t = np.arange(0, duration, 1.0/fs)
    y = np.ones_like(t) * p0
    # adjust params when bypass present
    if bypass:
        A = A * 0.6
        tau = tau * 0.6
        f0 = f0 * 1.2
    idx = t >= t0
    env = np.exp(-(t - t0)/tau) * idx
    y += A * env * np.sin(2 * np.pi * f0 * (t - t0))
    # add small ringing at higher freq
    y += 0.02 * env * np.sin(2 * np.pi * 200 * (t - t0))
    # add noise
    y += np.random.normal(0, noise_std, size=y.shape)
    return t, y


def extract_features(t, p, fs):
    """
    Compute a set of time, spectral and wavelet features from the transient.
    Returns a dict of features.
    """
    features = {}
    # time domain
    p0 = np.median(p[:int(0.1*len(p))])
    features['baseline'] = float(p0)
    peak_idx = np.argmax(p)
    features['peak_amp'] = float(p[peak_idx] - p0)
    # protect when t shape
    try:
        features['t_peak'] = float(t[peak_idx])
    except Exception:
        features['t_peak'] = 0.0
    # decay estimation: fit envelope to an exponential after peak
    try:
        idx_fit = (t >= t[peak_idx]) & (t <= t[peak_idx] + 1.0)
        if np.sum(idx_fit) > 10:
            env = np.abs(p[idx_fit] - p0)
            # avoid zeros
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
    try:
        yf = np.abs(rfft(p - p0))
        xf = rfftfreq(N, 1.0/fs)
    except Exception:
        yf = np.array([0.0])
        xf = np.array([0.0])
    # total energy and band energies
    total_energy = np.sum(yf**2)
    features['energy_total'] = float(total_energy)
    bands = [(0,20),(20,100),(100,500),(500,1000)]
    for i,(a,b) in enumerate(bands):
        idxb = (xf >= a) & (xf < b)
        features[f'energy_band_{a}_{b}'] = float(np.sum(yf[idxb]**2))
    # dominant freq
    if len(xf)>0:
        dom = xf[np.argmax(yf)] if np.sum(yf)>0 else 0.0
    else:
        dom = 0.0
    features['dom_freq'] = float(dom)
    # wavelet: use small set of coefficients
    try:
        coeffs = pywt.wavedec(p - p0, 'db4', level=4)
        # take energy of each level
        for i, c in enumerate(coeffs):
            features[f'wavelet_E_{i}'] = float(np.sum(np.array(c)**2))
    except Exception:
        for i in range(5):
            features[f'wavelet_E_{i}'] = 0.0
    return features


# ------------------- Synthetic dataset generation and ML -------------------

def generate_dataset(n_samples=4000, fs=2000, duration=5.0, seed=42):
    X = []
    y = []
    np.random.seed(seed)
    for i in range(n_samples):
        bypass = np.random.rand() < 0.5
        # sample random parameters
        p0 = np.random.uniform(1.8, 3.5)
        A = np.random.uniform(0.3, 1.0)
        f0 = np.random.uniform(10, 70)
        tau = np.random.uniform(0.15, 0.7)
        noise = np.random.uniform(0.005, 0.02)
        t0 = np.random.uniform(0.2, 0.8)
        t, p = generate_transient(duration=duration, fs=fs, p0=p0, t0=t0, A=A,
                                  f0=f0, tau=tau, noise_std=noise, bypass=bypass)
        feats = extract_features(t, p, fs)
        X.append([feats[k] for k in sorted(feats.keys())])
        y.append(int(bypass))
    feature_names = sorted(feats.keys())
    return np.array(X), np.array(y), feature_names


# ------------------- GUI Application -------------------
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax_time = fig.add_subplot(211)
        self.ax_spec = fig.add_subplot(212)
        super().__init__(fig)
        fig.tight_layout()


class WaterHammerGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('WaterHammer Simulator & Presentation Tool')
        self.setMinimumSize(1200, 700)
        # central widget
        w = QtWidgets.QWidget()
        self.setCentralWidget(w)
        layout = QtWidgets.QHBoxLayout()
        w.setLayout(layout)
        # left: controls
        ctrl = QtWidgets.QFrame()
        ctrl.setMinimumWidth(320)
        ctrl_layout = QtWidgets.QVBoxLayout()
        ctrl.setLayout(ctrl_layout)
        layout.addWidget(ctrl)
        # center: plots
        self.canvas = MplCanvas(self, width=6, height=5, dpi=100)
        layout.addWidget(self.canvas, stretch=1)
        # right: features and ML
        right = QtWidgets.QFrame()
        right.setMinimumWidth(320)
        right_layout = QtWidgets.QVBoxLayout()
        right.setLayout(right_layout)
        layout.addWidget(right)
        # Controls
        self.duration_spin = QtWidgets.QDoubleSpinBox(); self.duration_spin.setRange(1.0, 20.0); self.duration_spin.setValue(5.0); self.duration_spin.setSuffix(' s')
        self.fs_spin = QtWidgets.QSpinBox(); self.fs_spin.setRange(200, 5000); self.fs_spin.setValue(2000)
        self.p0_spin = QtWidgets.QDoubleSpinBox(); self.p0_spin.setRange(0.1, 10.0); self.p0_spin.setValue(2.5)
        self.amp_spin = QtWidgets.QDoubleSpinBox(); self.amp_spin.setRange(0.0, 2.0); self.amp_spin.setValue(0.6)
        self.f0_spin = QtWidgets.QDoubleSpinBox(); self.f0_spin.setRange(1,500); self.f0_spin.setValue(25)
        self.tau_spin = QtWidgets.QDoubleSpinBox(); self.tau_spin.setRange(0.01, 5.0); self.tau_spin.setValue(0.4)
        self.t0_spin = QtWidgets.QDoubleSpinBox(); self.t0_spin.setRange(0.0, 2.0); self.t0_spin.setValue(0.5)
        self.noise_spin = QtWidgets.QDoubleSpinBox(); self.noise_spin.setRange(0.0, 0.1); self.noise_spin.setSingleStep(0.001); self.noise_spin.setValue(0.01)
        self.bypass_check = QtWidgets.QCheckBox('Simular bypass (conexión clandestina)')
        btn_sim = QtWidgets.QPushButton('Simular')
        btn_save = QtWidgets.QPushButton('Guardar señal a CSV')
        btn_load = QtWidgets.QPushButton('Cargar CSV (t,p)')
        ctrl_layout.addWidget(QtWidgets.QLabel('<b>Parámetros de simulación</b>'))
        ctrl_layout.addWidget(QtWidgets.QLabel('Duración:')); ctrl_layout.addWidget(self.duration_spin)
        ctrl_layout.addWidget(QtWidgets.QLabel('Frecuencia de muestreo:')); ctrl_layout.addWidget(self.fs_spin)
        ctrl_layout.addWidget(QtWidgets.QLabel('Presión base (bar):')); ctrl_layout.addWidget(self.p0_spin)
        ctrl_layout.addWidget(QtWidgets.QLabel('Amplitud transiente (A):')); ctrl_layout.addWidget(self.amp_spin)
        ctrl_layout.addWidget(QtWidgets.QLabel('Frecuencia dominante (Hz):')); ctrl_layout.addWidget(self.f0_spin)
        ctrl_layout.addWidget(QtWidgets.QLabel('Constante de decaimiento (tau, s):')); ctrl_layout.addWidget(self.tau_spin)
        ctrl_layout.addWidget(QtWidgets.QLabel('Tiempo de inicio t0 (s):')); ctrl_layout.addWidget(self.t0_spin)
        ctrl_layout.addWidget(QtWidgets.QLabel('Ruido (std):')); ctrl_layout.addWidget(self.noise_spin)
        ctrl_layout.addWidget(self.bypass_check)
        ctrl_layout.addWidget(btn_sim)
        ctrl_layout.addStretch()
        ctrl_layout.addWidget(btn_save); ctrl_layout.addWidget(btn_load)
        # Right side: features and ML
        right_layout.addWidget(QtWidgets.QLabel('<b>Características extraídas</b>'))
        self.features_text = QtWidgets.QPlainTextEdit(); self.features_text.setReadOnly(True)
        right_layout.addWidget(self.features_text, stretch=1)
        # ML controls
        right_layout.addWidget(QtWidgets.QLabel('<b>Machine Learning (sintético)</b>'))
        # Training parameters
        right_layout.addWidget(QtWidgets.QLabel('Tamaño del dataset:'))
        self.n_samples_spin = QtWidgets.QSpinBox()
        self.n_samples_spin.setRange(100, 10000)
        self.n_samples_spin.setValue(400)
        right_layout.addWidget(self.n_samples_spin)
        right_layout.addWidget(QtWidgets.QLabel('N° estimadores RF:'))
        self.n_estimators_spin = QtWidgets.QSpinBox()
        self.n_estimators_spin.setRange(10, 500)
        self.n_estimators_spin.setValue(100)
        right_layout.addWidget(self.n_estimators_spin)
        right_layout.addWidget(QtWidgets.QLabel('C para SVM:'))
        self.svm_c_spin = QtWidgets.QDoubleSpinBox()
        self.svm_c_spin.setRange(0.1, 10.0)
        self.svm_c_spin.setSingleStep(0.1)
        self.svm_c_spin.setValue(1.0)
        right_layout.addWidget(self.svm_c_spin)
        btn_gen = QtWidgets.QPushButton('Generar dataset sintético y entrenar')
        self.ml_report = QtWidgets.QPlainTextEdit(); self.ml_report.setReadOnly(True)
        btn_predict = QtWidgets.QPushButton('Predecir escenario actual')
        btn_save_model = QtWidgets.QPushButton('Guardar modelo (joblib)')
        btn_load_model = QtWidgets.QPushButton('Cargar modelo (joblib)')
        right_layout.addWidget(btn_gen)
        right_layout.addWidget(self.ml_report)
        right_layout.addWidget(btn_predict)
        right_layout.addWidget(btn_save_model)
        right_layout.addWidget(btn_load_model)
        right_layout.addStretch()
        # Status bar
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)
        # Connect signals
        btn_sim.clicked.connect(self.on_simulate)
        btn_save.clicked.connect(self.on_save_csv)
        btn_load.clicked.connect(self.on_load_csv)
        btn_gen.clicked.connect(self.on_generate_and_train)
        btn_predict.clicked.connect(self.on_predict)
        btn_save_model.clicked.connect(self.on_save_model)
        btn_load_model.clicked.connect(self.on_load_model)
        # Data holders
        self.current_t = None
        self.current_p = None
        self.current_fs = None
        self.feature_names = None
        self.rf_model = None
        self.svm_model = None
        self.scaler = None
        # Helpful tip text
        help_text = ("Este programa genera señales sintéticas de golpe de ariete para "
                     "facilitar la explicación del proyecto. Para pruebas reales, cargue "
                     "un CSV con columnas 't' y 'p' (tiempo en s, presión en bar).\n\n"
                     "Panel ML: genera un dataset sintético (parámetros aleatorios), entrena "
                     "un RandomForest y un SVM y muestra métricas. Útil para explicar cómo "
                     "se obtienen características y cómo los clasificadores discriminan.")
        self.status.showMessage('Listo. ')
        self.features_text.setPlainText('Parámetros y características aparecerán aquí.')
        self.ml_report.setPlainText(help_text)

    # ---------------- GUI callbacks ----------------
    def on_simulate(self):
        duration = float(self.duration_spin.value())
        fs = int(self.fs_spin.value())
        p0 = float(self.p0_spin.value())
        A = float(self.amp_spin.value())
        f0 = float(self.f0_spin.value())
        tau = float(self.tau_spin.value())
        t0 = float(self.t0_spin.value())
        noise = float(self.noise_spin.value())
        bypass = bool(self.bypass_check.isChecked())
        t, p = generate_transient(duration=duration, fs=fs, p0=p0, t0=t0, A=A,
                                  f0=f0, tau=tau, noise_std=noise, bypass=bypass)
        self.current_t = t
        self.current_p = p
        self.current_fs = fs
        self.update_plots()
        feats = extract_features(t, p, fs)
        self.feature_names = sorted(feats.keys())
        # show features nicely
        txt = f"Simulación: bypass={bypass}\n"
        for k in sorted(feats.keys()):
            txt += f"{k}: {feats[k]:.6f}\n"
        self.features_text.setPlainText(txt)
        self.status.showMessage('Simulación completa.')

    def on_save_csv(self):
        if self.current_t is None:
            QtWidgets.QMessageBox.warning(self, 'Error', 'No hay señal para guardar. Simule o cargue una señal primero.')
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Guardar CSV', '', 'CSV files (*.csv)')
        if not path:
            return
        data = np.column_stack([self.current_t, self.current_p])
        np.savetxt(path, data, delimiter=',', header='t,p', comments='')
        self.status.showMessage(f'Señal guardada en {path}')

    def on_load_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Abrir CSV', '', 'CSV files (*.csv)')
        if not path:
            return
        try:
            data = np.loadtxt(path, delimiter=',', skiprows=1)
        except Exception:
            data = np.loadtxt(path, delimiter=',')
        if data.shape[1] < 2:
            QtWidgets.QMessageBox.warning(self, 'Error', 'CSV debe tener al menos 2 columnas: t,p')
            return
        t = data[:,0]
        p = data[:,1]
        # infer fs
        dt = np.mean(np.diff(t))
        fs = int(round(1.0/dt)) if dt>0 else 1000
        self.current_t = t
        self.current_p = p
        self.current_fs = fs
        self.update_plots()
        feats = extract_features(t, p, fs)
        txt = f"Señal cargada: {os.path.basename(path)}\n"
        for k in sorted(feats.keys()):
            txt += f"{k}: {feats[k]:.6f}\n"
        self.features_text.setPlainText(txt)
        self.status.showMessage('CSV cargado y analizado.')

    def update_plots(self):
        t = self.current_t
        p = self.current_p
        if t is None:
            return
        self.canvas.ax_time.clear()
        self.canvas.ax_spec.clear()
        self.canvas.ax_time.plot(t, p)
        self.canvas.ax_time.set_ylabel('Presión (bar)')
        self.canvas.ax_time.set_xlabel('Tiempo (s)')
        # spectrogram
        fs = self.current_fs
        f, Pxx = signal.welch(p - np.median(p[:max(1,int(0.05*len(p)))]), fs=fs, nperseg=1024)
        self.canvas.ax_spec.semilogy(f, Pxx)
        self.canvas.ax_spec.set_xlabel('Freq (Hz)')
        self.canvas.ax_spec.set_ylabel('PSD')
        self.canvas.figure.tight_layout()
        self.canvas.draw()

    def on_generate_and_train(self):
        n_samples = int(self.n_samples_spin.value())
        n_estimators = int(self.n_estimators_spin.value())
        svm_c = float(self.svm_c_spin.value())
        self.status.showMessage('Generando dataset sintético...')
        QtWidgets.QApplication.processEvents()
        X, y, feature_names = generate_dataset(n_samples=n_samples, fs=int(self.fs_spin.value()), duration=float(self.duration_spin.value()))
        self.status.showMessage('Entrenando modelos...')
        # Escalado: entrenamos sobre features estandarizadas para consistencia con trainer externo
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(Xs, y, test_size=0.25, random_state=42)
        rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        rf.fit(X_train, y_train)
        svm = SVC(kernel='rbf', probability=True, C=svm_c)
        svm.fit(X_train, y_train)
        ypred_rf = rf.predict(X_test)
        ypred_svm = svm.predict(X_test)
        rep = 'RandomForest\n'
        rep += f"Accuracy: {accuracy_score(y_test, ypred_rf):.4f}\n"
        rep += classification_report(y_test, ypred_rf)
        rep += '\nSVM\n'
        rep += f"Accuracy: {accuracy_score(y_test, ypred_svm):.4f}\n"
        rep += classification_report(y_test, ypred_svm)
        self.rf_model = rf
        self.svm_model = svm
        self.scaler = scaler
        self.ml_report.setPlainText(rep)
        self.feature_names = feature_names
        self.status.showMessage('Modelos entrenados.')

    def on_predict(self):
        if self.current_t is None:
            QtWidgets.QMessageBox.warning(self, 'Error', 'No hay señal actual para predecir. Simule o cargue una señal.')
            return
        if self.rf_model is None and self.svm_model is None:
            QtWidgets.QMessageBox.warning(self, 'Error', 'No hay modelo entrenado o cargado.')
            return

        feats = extract_features(self.current_t, self.current_p, self.current_fs)

        # construir X en el orden correcto: preferir feature_names guardadas
        if self.feature_names is not None:
            try:
                X = np.array([feats[k] for k in self.feature_names]).reshape(1, -1)
            except KeyError as e:
                QtWidgets.QMessageBox.warning(self, 'Error', f'La característica {e} no se encontró en la señal. Verifique feature_names.')
                return
        else:
            # caida segura: usar orden alfabético (comportamiento previo)
            X = np.array([feats[k] for k in sorted(feats.keys())]).reshape(1, -1)

        # aplicar escalado si hay scaler guardado
        Xs = X
        if hasattr(self, 'scaler') and self.scaler is not None:
            try:
                Xs = self.scaler.transform(X)
            except Exception as e:
                # informar pero intentar predecir con X original
                self.status.showMessage(f'Warning: fallo al aplicar scaler: {e} — usando features sin escalar.')
                Xs = X

        out_text = self.features_text.toPlainText() + '\n\nPredicciones:\n'
        # RF
        if self.rf_model is not None:
            try:
                if hasattr(self.rf_model, 'predict_proba'):
                    prf = self.rf_model.predict_proba(Xs)[0]
                    ps = self.rf_model.predict(Xs)[0]
                    out_text += f"RandomForest: clase={ps} (prob_no_bypass={prf[0]:.3f}, prob_bypass={prf[1]:.3f})\n"
                else:
                    ps = self.rf_model.predict(Xs)[0]
                    out_text += f"RandomForest: clase={ps}\n"
            except Exception as e:
                out_text += f"RandomForest: error en predicción: {e}\n"

        # SVM
        if self.svm_model is not None:
            try:
                ps_svm = self.svm_model.predict(Xs)[0]
                out_text += f"SVM: clase={ps_svm}\n"
            except Exception as e:
                out_text += f"SVM: error en predicción: {e}\n"

        self.features_text.setPlainText(out_text)
        self.status.showMessage('Predicción realizada.')

    def on_save_model(self):
        if self.rf_model is None and self.svm_model is None:
            QtWidgets.QMessageBox.warning(self, 'Error', 'No hay modelo entrenado para guardar.')
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Guardar modelo', '', 'Joblib files (*.joblib)')
        if not path:
            return
        # Guardar en formato compatible tanto con el programa principal como con el trainer externo
        payload = {
            'rf': self.rf_model,
            'svm': self.svm_model,
            'feature_names': self.feature_names,
            'scaler': self.scaler
        }
        try:
            joblib.dump(payload, path)
            self.status.showMessage(f'Modelo guardado en {path}')
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'Error', f'No se pudo guardar el modelo: {e}')

    def on_load_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Cargar modelo', '', 'Joblib files (*.joblib)')
        if not path:
            return
        try:
            data = joblib.load(path)
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, 'Error', f'No se pudo cargar el archivo.\n{e}')
            return

        # soportar distintos formatos:
        # - formato antiguo/programa principal: {'rf':..., 'svm':..., 'feature_names':...}
        # - formato del trainer: {'models': {'rf':..., 'svm':...}, 'scaler':..., 'feature_names':...}
        rf = None
        svm = None
        scaler = None
        feature_names = None

        if isinstance(data, dict):
            # caso entrenador que guardó 'models' dict
            if 'models' in data and isinstance(data['models'], dict):
                rf = data['models'].get('rf', None)
                svm = data['models'].get('svm', None)
            # claves directas
            rf = rf or data.get('rf', None)
            svm = svm or data.get('svm', None)
            scaler = data.get('scaler', None)
            feature_names = data.get('feature_names', None)
        else:
            QtWidgets.QMessageBox.warning(self, 'Error', 'Formato de archivo no reconocido.')
            return

        if rf is None and svm is None:
            QtWidgets.QMessageBox.warning(self, 'Error', 'El archivo no contiene modelos "rf" ni "svm".')
            return

        # asignar a la GUI
        self.rf_model = rf
        self.svm_model = svm
        self.scaler = scaler
        self.feature_names = feature_names

        self.ml_report.setPlainText('Modelo cargado exitosamente.\n\nPara predecir, cargue una señal y presione "Predecir".')
        self.status.showMessage(f'Modelo cargado desde {path}')

# ------------------- Main -------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    gui = WaterHammerGUI()
    gui.show()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
