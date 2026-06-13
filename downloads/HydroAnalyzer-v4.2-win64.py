"""
HydroAnalyzer v4.6 — Análisis unificado de transientes hidráulicos
====================================================================


Dependencias: PyQt5, numpy, scipy, scikit-learn, pywavelets,
              matplotlib, joblib, pandas.
"""

from __future__ import annotations

import os
import sys
import time
import datetime
import traceback
import multiprocessing
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy.signal as sp_signal
from scipy.fft import rfft, rfftfreq
from scipy.optimize import curve_fit

import pywt
import joblib

import matplotlib
matplotlib.use("Qt5Agg")
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from PyQt5 import QtCore, QtGui, QtWidgets

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Gradient boosting opcional. Si las librerías no están instaladas,
# las opciones correspondientes en la UI quedan deshabilitadas pero
# el resto del programa sigue funcionando.
try:
    from xgboost import XGBClassifier  # type: ignore
    XGBOOST_AVAILABLE = True
except Exception:
    XGBClassifier = None  # type: ignore
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier  # type: ignore
    LIGHTGBM_AVAILABLE = True
except Exception:
    LGBMClassifier = None  # type: ignore
    LIGHTGBM_AVAILABLE = False


# ============================================================================
# 1. CONFIGURACIÓN Y TEMA
# ============================================================================

APP_NAME = "HydroAnalyzer"
APP_VERSION = "4.6"

# ── Soporte para ejecutables congelados (PyInstaller / py2exe) ─────────
# sys.frozen lo define el bootloader del empaquetador. En un .exe
# congelado, los procesos hijos que lanza joblib/loky (backend de
# scikit-learn para n_jobs con PROCESOS) se crean RE-EJECUTANDO el
# ejecutable completo: sin protección, cada "worker" abre una ventana
# nueva de HydroAnalyzer, el entrenamiento se bloquea en la calibración
# y cerrar una de esas ventanas produce TerminatedWorkerError.
#
# Defensa en dos capas:
#   1) multiprocessing.freeze_support() como PRIMERA línea del bloque
#      __main__ (al final del archivo): intercepta los relanzamientos
#      hijos antes de que ejecuten main().
#   2) PROC_N_JOBS: los componentes de sklearn que paralelizan con
#      PROCESOS (CalibratedClassifierCV, cross_val_score) usan 1 hilo
#      (backend secuencial → cero subprocesos) cuando IS_FROZEN.
#      RandomForest, XGBoost y LightGBM conservan n_jobs=-1 porque
#      paralelizan con HILOS (sklearn usa prefer="threads" en bosques;
#      XGB/LGBM usan OpenMP nativo) — seguros y rápidos en el .exe.
IS_FROZEN = bool(getattr(sys, "frozen", False))
PROC_N_JOBS = 1 if IS_FROZEN else -1

# Filtro de archivos para diálogos de apertura: aceptamos cualquier
# archivo de texto plano. El parser load_csv_signal es tolerante a
# basura no numérica y soporta CSV, TXT, LOG, etc.
SIGNAL_FILE_FILTER = (
    "Archivos de señal (*.csv *.txt *.log *.dat *.tsv);;"
    "CSV (*.csv);;"
    "Texto (*.txt *.log *.dat *.tsv);;"
    "Todos los archivos (*)"
)
SIGNAL_SAVE_FILTER = "CSV (*.csv);;Texto (*.txt);;Todos (*)"
MODEL_FORMAT_VERSION = 4   # incremento por adición de XGB/LGBM/calibración

# ────────────────────────────────────────────────────────────────────────────
# Registro central de modelos disponibles.
# Cada clave identifica unívocamente al modelo en TrainingResult.models, en la
# UI, en los .joblib serializados y en la tabla de detalle por muestra.
#
# La pseudo-clave 'ensemble' representa el voting agregado calculado en
# tiempo de predicción a partir de los modelos disponibles. NO es un modelo
# entrenado, así que no aparece en la lista canónica de modelos serializables.
# ────────────────────────────────────────────────────────────────────────────
MODEL_KEYS: List[str] = ["rf", "svm", "xgb", "lgbm"]
ENSEMBLE_KEY: str = "ensemble"

MODEL_DISPLAY_NAMES: Dict[str, str] = {
    "rf":       "Random Forest",
    "svm":      "SVM (RBF)",
    "xgb":      "XGBoost",
    "lgbm":     "LightGBM",
    "ensemble": "Voting Ensemble",
}
MODEL_SHORT_NAMES: Dict[str, str] = {
    "rf":       "RF",
    "svm":      "SVM",
    "xgb":      "XGB",
    "lgbm":     "LGBM",
    "ensemble": "ENS",
}

def model_is_available(key: str) -> bool:
    """¿Está la dependencia para entrenar este modelo instalada?"""
    if key == "xgb":
        return XGBOOST_AVAILABLE
    if key == "lgbm":
        return LIGHTGBM_AVAILABLE
    return True   # rf y svm siempre vienen con scikit-learn

# Paleta (Tokyo Night style)
COLOR_BG        = "#1a1b26"
COLOR_PANEL     = "#24283b"
COLOR_PANEL_ALT = "#1f2335"
COLOR_BORDER    = "#3b4261"
COLOR_TEXT      = "#c0caf5"
COLOR_TEXT_DIM  = "#9aa5ce"
COLOR_ACCENT    = "#7aa2f7"
COLOR_SUCCESS   = "#9ece6a"
COLOR_WARNING   = "#e0af68"
COLOR_DANGER    = "#f7768e"
COLOR_CYAN      = "#7dcfff"
COLOR_MAGENTA   = "#bb9af7"
COLOR_ORANGE    = "#ff9e64"

FREQ_BANDS: List[Tuple[int, int]] = [(0, 20), (20, 100), (100, 500), (500, 1000)]
WAVELET_NAME = "db4"
WAVELET_LEVEL = 4

AUTHOR_WEBSITE = "https://andyrcc.github.io/"
PROJECT_WEBSITE = "https://andyrcc.github.io/HydraDetect-AI/"


# ============================================================================
# 1.5. EFECTOS VISUALES Y ANIMACIONES (v4.0 «Aurora»)
# ============================================================================

class FX:
    """
    Helpers estáticos de animación. Todas las animaciones se guardan como
    atributo del widget animado (``widget._fx_anims``) para evitar que el
    garbage collector las mate a mitad de camino — un bug clásico de PyQt.

    Notas de diseño:
      * Un widget solo puede tener UN QGraphicsEffect a la vez. Por eso
        ``fade_in`` ELIMINA el efecto al terminar (si no, el opacity effect
        rompería cosas como tooltips y drop shadows posteriores).
      * Todas las curvas usan easing OutCubic — rápido al inicio, suave al
        final: la firma de una UI moderna.

    Ciclo de vida (v4.1 — fix del crash «QPropertyAnimation has been
    deleted»):
      * Las animaciones one-shot se destruyen con ``finished →
        deleteLater``. Eso borra el objeto C++ pero el wrapper Python
        puede seguir dentro de ``widget._fx_anims`` hasta la siguiente
        purga. Tocar un wrapper muerto lanza RuntimeError y, como PyQt5
        aborta el proceso ante excepciones no capturadas en slots, la app
        entera crasheaba. Por eso TODA inspección de animaciones
        almacenadas pasa por ``_is_running`` (que captura RuntimeError) y
        todos los puntos de entrada públicos son a prueba de widgets en
        destrucción.
    """

    DURATION_FAST = 180
    DURATION_MED  = 320
    DURATION_SLOW = 550

    # ------------------------------------------------------------------
    @staticmethod
    def _is_running(anim) -> bool:
        """True solo si la animación sigue viva (C++ no destruido) y en
        ejecución. Nunca lanza: un wrapper muerto cuenta como 'no'."""
        try:
            return anim.state() == QtCore.QAbstractAnimation.Running
        except RuntimeError:
            return False    # objeto C++ ya eliminado (deleteLater)

    @staticmethod
    def _keep(widget, anim):
        """
        Evita garbage-collection de la animación guardándola en el widget.
        La purga de animaciones previas es a prueba de wrappers muertos:
        este era el origen del crash de v4.0 (RuntimeError al llamar
        .state() sobre una animación cuyo C++ ya había sido destruido).
        """
        try:
            prev = list(getattr(widget, "_fx_anims", []))
        except RuntimeError:
            return    # el propio widget está siendo destruido
        alive = [a for a in prev if FX._is_running(a)]
        alive.append(anim)
        try:
            widget._fx_anims = alive
        except RuntimeError:
            pass

    @staticmethod
    def _start_oneshot(widget, anim):
        """Arranca una animación one-shot con auto-destrucción segura."""
        FX._keep(widget, anim)
        # deleteLater (en vez de DeleteWhenStopped) + purga blindada:
        # el C++ se libera, y los wrappers muertos que queden en
        # _fx_anims son ignorados por _is_running.
        anim.finished.connect(anim.deleteLater)
        anim.start()

    # ------------------------------------------------------------------
    @staticmethod
    def fade_in(widget: QtWidgets.QWidget, duration: int = None,
                start: float = 0.0):
        """Fade-in con easing; elimina el efecto al terminar."""
        if widget is None:
            return
        duration = duration or FX.DURATION_MED
        try:
            eff = QtWidgets.QGraphicsOpacityEffect(widget)
            widget.setGraphicsEffect(eff)
            anim = QtCore.QPropertyAnimation(eff, b"opacity", widget)
        except RuntimeError:
            return    # widget en proceso de destrucción
        anim.setDuration(duration)
        anim.setStartValue(start)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)

        def _cleanup():
            # Quitar el efecto restaura el render normal (tooltips, shadows…)
            try:
                widget.setGraphicsEffect(None)
            except RuntimeError:
                pass  # widget destruido durante la animación
        anim.finished.connect(_cleanup)
        FX._start_oneshot(widget, anim)

    # ------------------------------------------------------------------
    @staticmethod
    def slide_fade_in(widget: QtWidgets.QWidget, dy: int = 16,
                      duration: int = None):
        """Aparece deslizando hacia arriba + fade simultáneo."""
        if widget is None:
            return
        try:
            visible = widget.isVisible()
        except RuntimeError:
            return
        if not visible:
            FX.fade_in(widget, duration)
            return
        duration = duration or FX.DURATION_MED
        try:
            # geometría: animar pos desde (x, y+dy) hasta (x, y)
            end_pos = widget.pos()
            start_pos = QtCore.QPoint(end_pos.x(), end_pos.y() + dy)
            anim_pos = QtCore.QPropertyAnimation(widget, b"pos", widget)
        except RuntimeError:
            return
        anim_pos.setDuration(duration)
        anim_pos.setStartValue(start_pos)
        anim_pos.setEndValue(end_pos)
        anim_pos.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        FX._start_oneshot(widget, anim_pos)
        FX.fade_in(widget, duration)

    # ------------------------------------------------------------------
    @staticmethod
    def glow(widget: QtWidgets.QWidget, color: str,
             blur: int = 26, alpha: int = 170):
        """Aplica un drop-shadow tipo glow del color dado (persistente)."""
        if widget is None:
            return None
        try:
            eff = QtWidgets.QGraphicsDropShadowEffect(widget)
            c = QtGui.QColor(color); c.setAlpha(alpha)
            eff.setColor(c)
            eff.setBlurRadius(blur)
            eff.setOffset(0, 0)
            widget.setGraphicsEffect(eff)
        except RuntimeError:
            return None    # widget en destrucción
        return eff

    # ------------------------------------------------------------------
    @staticmethod
    def pulse_glow(widget: QtWidgets.QWidget, color: str,
                   blur_min: int = 8, blur_max: int = 30,
                   period_ms: int = 1600):
        """Glow que respira en bucle infinito (para CTAs importantes)."""
        eff = FX.glow(widget, color, blur=blur_min)
        if eff is None:
            return None
        try:
            anim = QtCore.QPropertyAnimation(eff, b"blurRadius", widget)
        except RuntimeError:
            return None
        anim.setDuration(period_ms)
        anim.setStartValue(blur_min)
        anim.setKeyValueAt(0.5, blur_max)
        anim.setEndValue(blur_min)
        anim.setEasingCurve(QtCore.QEasingCurve.InOutSine)
        anim.setLoopCount(-1)   # infinito — nunca emite finished
        FX._keep(widget, anim)
        anim.start()
        return anim

    # ------------------------------------------------------------------
    @staticmethod
    def animate_progress(bar: QtWidgets.QProgressBar, target: int,
                         duration: int = 240):
        """Mueve la barra de progreso suavemente hasta `target`."""
        if bar is None:
            return
        try:
            cur = bar.value()
        except RuntimeError:
            return
        target = int(target)
        if cur == target:
            return
        # Si hay una animación de progreso anterior aún corriendo en esta
        # barra, detenerla: dos animaciones de `value` simultáneas
        # pelearían entre sí y la barra "temblaría".
        prev = getattr(bar, "_fx_progress_anim", None)
        if prev is not None and FX._is_running(prev):
            try:
                prev.stop()
            except RuntimeError:
                pass
        try:
            anim = QtCore.QPropertyAnimation(bar, b"value", bar)
        except RuntimeError:
            return
        anim.setDuration(duration)
        anim.setStartValue(cur)
        anim.setEndValue(target)
        anim.setEasingCurve(QtCore.QEasingCurve.OutCubic)
        bar._fx_progress_anim = anim
        FX._start_oneshot(bar, anim)

    # ------------------------------------------------------------------
    @staticmethod
    def attach_tab_fade(tabs: QtWidgets.QTabWidget,
                        duration: int = None):
        """
        Conecta un fade-in del contenido cada vez que cambia la pestaña
        activa de `tabs`. Idempotente (no se conecta dos veces).
        """
        if tabs is None or getattr(tabs, "_fx_fade_attached", False):
            return
        tabs._fx_fade_attached = True
        d = duration or FX.DURATION_FAST

        def _on_changed(idx: int):
            try:
                w = tabs.widget(idx)
            except RuntimeError:
                return    # el QTabWidget está siendo destruido (cierre)
            if w is not None:
                FX.fade_in(w, d)
        tabs.currentChanged.connect(_on_changed)


class Plot3DCanvas(FigureCanvas):
    """
    Canvas matplotlib dedicado a un único axes 3D. Separado de PlotCanvas
    porque la proyección 3D necesita crearse con add_subplot(projection=…)
    y el ciclo limpiar/redibujar es distinto al de los plots 2D.
    """
    def __init__(self):
        self.fig = Figure(facecolor=COLOR_PANEL, tight_layout=True)
        super().__init__(self.fig)
        # (v4.2) Política Expanding explícita: garantiza que el canvas
        # llene todo el espacio disponible (algunas versiones de
        # matplotlib dejan la política en Preferred, lo que en pantalla
        # completa hacía que otros widgets absorbieran el espacio).
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self.ax3d = None
        self._make_axes()

    def _make_axes(self):
        self.fig.clf()
        self.ax3d = self.fig.add_subplot(111, projection="3d")
        self.ax3d.set_facecolor(COLOR_PANEL)
        # Paneles laterales translúcidos para que el dark theme respire
        try:
            for pane in (self.ax3d.xaxis.pane, self.ax3d.yaxis.pane,
                         self.ax3d.zaxis.pane):
                pane.set_facecolor(COLOR_PANEL_ALT)
                pane.set_edgecolor(COLOR_BORDER)
                pane.set_alpha(0.35)
        except Exception:
            pass
        for axis in (self.ax3d.xaxis, self.ax3d.yaxis, self.ax3d.zaxis):
            axis.label.set_color(COLOR_TEXT_DIM)
            axis.set_tick_params(colors=COLOR_TEXT_DIM, labelsize=7)

    def show_empty(self, msg: str = "Genera o carga una señal"):
        self._make_axes()
        self.ax3d.text2D(0.5, 0.5, msg, ha="center", va="center",
                         transform=self.ax3d.transAxes,
                         color=COLOR_TEXT_DIM, fontsize=11)
        self.ax3d.set_xticks([]); self.ax3d.set_yticks([])
        self.ax3d.set_zticks([])
        self.draw_idle()

    def plot_spectrogram_surface(self, t: np.ndarray, p: np.ndarray,
                                  fs: int, baseline: float,
                                  title_suffix: str = ""):
        """
        Superficie 3D (tiempo × frecuencia × dB) del espectrograma.
        Downsamplea a ~110×110 celdas para mantener el render fluido
        incluso con señales largas.
        """
        self._make_axes()
        nperseg = min(256, max(64, len(p) // 32))
        f_sg, t_sg, Sxx = sp_signal.spectrogram(
            p - baseline, fs=fs, nperseg=nperseg
        )
        Z = 10 * np.log10(Sxx + 1e-12)

        # Limitar a frecuencias útiles (igual que el espectrograma 2D)
        f_max = min(fs / 2, 500)
        f_mask = f_sg <= f_max
        f_sg = f_sg[f_mask]; Z = Z[f_mask, :]

        # Downsampling para fluidez
        def decimate(arr, axis, max_n):
            n = arr.shape[axis]
            if n <= max_n:
                return arr, 1
            step = int(np.ceil(n / max_n))
            sl = [slice(None)] * arr.ndim
            sl[axis] = slice(None, None, step)
            return arr[tuple(sl)], step

        Z, step_f = decimate(Z, 0, 110)
        Z, step_t = decimate(Z, 1, 110)
        f_ds = f_sg[::step_f]
        t_ds = t_sg[::step_t]

        T, F = np.meshgrid(t_ds, f_ds)
        surf = self.ax3d.plot_surface(
            T, F, Z, cmap="magma",
            linewidth=0, antialiased=True,
            rstride=1, cstride=1, alpha=0.96,
        )
        self.ax3d.set_xlabel("Tiempo (s)", fontsize=8, labelpad=6)
        self.ax3d.set_ylabel("Frecuencia (Hz)", fontsize=8, labelpad=6)
        self.ax3d.set_zlabel("Potencia (dB)", fontsize=8, labelpad=4)
        title = "Espectrograma 3D"
        if title_suffix:
            title += f" — {title_suffix}"
        self.ax3d.set_title(title, fontsize=10, pad=10, color=COLOR_TEXT)
        # Vista inicial agradable
        self.ax3d.view_init(elev=32, azim=-58)
        cb = self.fig.colorbar(surf, ax=self.ax3d, pad=0.08, shrink=0.7)
        cb.ax.tick_params(colors=COLOR_TEXT_DIM, labelsize=7)
        cb.outline.set_edgecolor(COLOR_BORDER)
        self.draw_idle()


class AuroraSplash(QtWidgets.QSplashScreen):
    """
    Splash screen pintado a mano: degradado vertical estilo aurora,
    nombre de la app, versión y tagline. Se desvanece al cerrar.
    """
    W, H = 560, 320

    def __init__(self):
        pix = QtGui.QPixmap(self.W, self.H)
        pix.fill(QtCore.Qt.transparent)
        painter = QtGui.QPainter(pix)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)

        # Fondo redondeado con degradado
        grad = QtGui.QLinearGradient(0, 0, self.W, self.H)
        grad.setColorAt(0.0, QtGui.QColor("#16161e"))
        grad.setColorAt(0.45, QtGui.QColor("#1f2335"))
        grad.setColorAt(1.0, QtGui.QColor("#24283b"))
        path = QtGui.QPainterPath()
        path.addRoundedRect(QtCore.QRectF(0, 0, self.W, self.H), 18, 18)
        painter.fillPath(path, QtGui.QBrush(grad))

        # «Aurora»: bandas de color translúcidas en diagonal
        for i, (col, y0) in enumerate([
            (COLOR_ACCENT,  40), (COLOR_CYAN, 110),
            (COLOR_MAGENTA, 180), (COLOR_SUCCESS, 250),
        ]):
            band = QtGui.QLinearGradient(0, y0, self.W, y0 + 70)
            c1 = QtGui.QColor(col); c1.setAlpha(0)
            c2 = QtGui.QColor(col); c2.setAlpha(46)
            band.setColorAt(0.0, c1); band.setColorAt(0.5, c2)
            band.setColorAt(1.0, c1)
            painter.save()
            painter.setClipPath(path)
            painter.fillRect(QtCore.QRectF(-60, y0, self.W + 120, 70),
                             QtGui.QBrush(band))
            painter.restore()

        # Borde sutil
        pen = QtGui.QPen(QtGui.QColor(COLOR_BORDER)); pen.setWidth(1)
        painter.setPen(pen)
        painter.drawPath(path)

        # Texto
        painter.setPen(QtGui.QColor(COLOR_ACCENT))
        f = QtGui.QFont("Segoe UI", 30, QtGui.QFont.Bold)
        painter.setFont(f)
        painter.drawText(QtCore.QRectF(0, 86, self.W, 60),
                         QtCore.Qt.AlignCenter, "💧 HydroAnalyzer")
        painter.setPen(QtGui.QColor(COLOR_TEXT))
        painter.setFont(QtGui.QFont("Segoe UI", 11))
        painter.drawText(QtCore.QRectF(0, 150, self.W, 30),
                         QtCore.Qt.AlignCenter,
                         "Análisis y clasificación ML de transientes hidráulicos")
        painter.setPen(QtGui.QColor(COLOR_TEXT_DIM))
        painter.setFont(QtGui.QFont("Segoe UI", 9))
        painter.drawText(QtCore.QRectF(0, 256, self.W, 24),
                         QtCore.Qt.AlignCenter,
                         f"v{APP_VERSION} «Aurora»  ·  {AUTHOR_WEBSITE}")
        painter.end()

        super().__init__(pix)
        self.setWindowFlags(QtCore.Qt.SplashScreen
                            | QtCore.Qt.FramelessWindowHint
                            | QtCore.Qt.WindowStaysOnTopHint)

    def fade_out_and_close(self, duration: int = 420):
        anim = QtCore.QPropertyAnimation(self, b"windowOpacity", self)
        anim.setDuration(duration)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.setEasingCurve(QtCore.QEasingCurve.InCubic)
        anim.finished.connect(self.close)
        self._fade_anim = anim   # mantener referencia
        anim.start()


class ClickableLabel(QtWidgets.QLabel):
    """QLabel que emite `clicked` al hacer click izquierdo (v4.4) —
    usado para que el título «HydroAnalyzer» abra la página de
    créditos."""
    clicked = QtCore.pyqtSignal()

    def mousePressEvent(self, ev):
        if ev.button() == QtCore.Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(ev)


class CreditsPage(QtWidgets.QWidget):
    """
    Página de créditos a PANTALLA COMPLETA (v4.4) — reemplaza al antiguo
    diálogo emergente. Vive como página 1 del QStackedWidget central:
    nada de ventanas; se entra haciendo click en el título
    «HydroAnalyzer» (o ❤ Créditos / Ctrl+G) y se vuelve con el botón
    «← Volver» o con Esc.

    El fondo es un SISTEMA SOLAR animado pintado a mano con QPainter:
      • Campo de ~150 estrellas titilantes (12 brillantes con destello
        en cruz) sobre un gradiente de espacio profundo con nebulosas.
      • Sol central con núcleo, corona y halo en gradientes radiales,
        pulsando suavemente.
      • 8 planetas en órbitas elípticas con perspectiva (pseudo-3D):
        cuando sin(θ)<0 pasan POR DETRÁS del sol (se dibujan antes,
        más pequeños y tenues) y cuando sin(θ)>0 pasan por delante.
        Cada planeta se ilumina DESDE el sol (gradiente desplazado
        hacia el centro), deja una estela de movimiento, Saturno tiene
        anillos, Júpiter bandas y la Tierra una luna orbitándola.
      • Cometas periódicos cruzando con cola degradada.
    La animación corre a ~30 fps SOLO mientras la página está visible
    (showEvent/hideEvent arrancan/detienen el timer → costo cero en el
    uso normal del programa).
    """

    back_requested = QtCore.pyqtSignal()

    #            nombre      color      r_orb  r_px  periodo  extra
    PLANETS = [
        ("Mercurio", "#b9a89a", 0.135, 4.0,  4.6,  None),
        ("Venus",    "#e8c46a", 0.195, 7.0,  7.4,  None),
        ("Tierra",   "#5b9bd5", 0.270, 8.0, 10.0,  "moon"),
        ("Marte",    "#d1604d", 0.345, 6.0, 13.6,  None),
        ("Júpiter",  "#d9a066", 0.480, 17.0, 21.0, "bands"),
        ("Saturno",  "#e6cf9a", 0.615, 14.0, 29.0, "rings"),
        ("Urano",    "#8fd3e8", 0.735, 10.0, 38.0, None),
        ("Neptuno",  "#6f8fe8", 0.860, 10.0, 48.0, None),
    ]
    TILT = 0.36          # compresión vertical de las órbitas
    TRAIL_LEN = 24       # puntos de estela por planeta
    FPS_MS = 33          # ~30 fps

    def __init__(self, parent=None):
        super().__init__(parent)
        self._t0 = time.monotonic()
        self._rng = np.random.default_rng(7)
        self._stars: List[Dict[str, float]] = []
        self._trails: List[List[Tuple[float, float, float]]] = \
            [[] for _ in self.PLANETS]
        self._phases = [self._rng.uniform(0, 2 * np.pi)
                        for _ in self.PLANETS]
        self._comet: Optional[Dict[str, Any]] = None
        self._next_comet = time.monotonic() + 3.0
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update)

        self._build_overlay_ui()

        # Esc también vuelve al programa
        esc = QtWidgets.QShortcut(QtGui.QKeySequence("Esc"), self)
        esc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        esc.activated.connect(self.back_requested.emit)

    # ------------------------------------------------------------------
    def _build_overlay_ui(self):
        """Botón «Volver» + tarjeta glass con el contenido de créditos.
        Son hijos del widget → Qt los pinta ENCIMA del sistema solar."""
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 18)

        # Barra superior: volver
        top = QtWidgets.QHBoxLayout()
        self.btn_back = QtWidgets.QPushButton("←  Volver al programa")
        self.btn_back.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_back.setMinimumHeight(38)
        self.btn_back.setStyleSheet(f"""
            QPushButton {{
                background: rgba(13, 15, 26, 0.72); color: {COLOR_TEXT};
                border: 1px solid rgba(122, 162, 247, 0.45);
                border-radius: 9px; padding: 7px 16px; font-weight: 600;
            }}
            QPushButton:hover {{
                color: {COLOR_CYAN}; border-color: {COLOR_CYAN};
                background: rgba(18, 22, 38, 0.85);
            }}
        """)
        self.btn_back.clicked.connect(self.back_requested.emit)
        top.addWidget(self.btn_back, 0, QtCore.Qt.AlignLeft)
        top.addStretch(1)
        hint = QtWidgets.QLabel("Esc para volver")
        hint.setStyleSheet(
            f"color: rgba(169,177,214,0.55); font-size: 8pt;"
            f"background: transparent;"
        )
        top.addWidget(hint, 0, QtCore.Qt.AlignRight)
        outer.addLayout(top)

        outer.addStretch(12)   # (v4.4b) tarjeta más abajo → el sol queda
                               # visible en el tercio superior

        # Tarjeta glass central
        card = QtWidgets.QFrame()
        card.setObjectName("creditsCard")
        card.setMaximumWidth(620)
        card.setStyleSheet(f"""
            QFrame#creditsCard {{
                background: rgba(10, 12, 22, 0.78);
                border: 1px solid rgba(122, 162, 247, 0.40);
                border-radius: 18px;
            }}
            QLabel {{ background: transparent; color: {COLOR_TEXT}; }}
            QPushButton#webButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {COLOR_ACCENT}, stop:1 #5a7fd6);
                color: #0d0f1a; font-weight: 700;
                border: 1px solid {COLOR_ACCENT}; border-radius: 9px;
                padding: 10px 18px;
            }}
            QPushButton#webButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 {COLOR_CYAN}, stop:1 {COLOR_ACCENT});
                border-color: {COLOR_CYAN};
            }}
        """)
        cv = QtWidgets.QVBoxLayout(card)
        cv.setContentsMargins(34, 26, 34, 24); cv.setSpacing(12)

        big = QtWidgets.QLabel("💧 HydroAnalyzer")
        big.setStyleSheet(
            f"font-size: 24pt; font-weight: 800; color: {COLOR_ACCENT};"
            f"background: transparent;"
        )
        big.setAlignment(QtCore.Qt.AlignCenter)
        ver = QtWidgets.QLabel(f"v{APP_VERSION} «Aurora»")
        ver.setStyleSheet(f"font-size: 10pt; color: {COLOR_TEXT_DIM};"
                          f"background: transparent;")
        ver.setAlignment(QtCore.Qt.AlignCenter)
        cv.addWidget(big); cv.addWidget(ver)

        thanks = QtWidgets.QLabel(
            "<p style='font-size:11pt;'>"
            "<b>¡Gracias por usar HydroAnalyzer!</b></p>"
            "<p>Esta herramienta combina física de transientes "
            "hidráulicos con machine learning para detectar bypass en "
            "sistemas de tuberías: simulación, filtrado, entrenamiento, "
            "validación cruzada y análisis — todo en un solo lugar.</p>"
            "<p>Si te resulta útil, te invito a conocer más proyectos y "
            "novedades en mi página web:</p>"
        )
        thanks.setWordWrap(True)
        thanks.setAlignment(QtCore.Qt.AlignCenter)
        cv.addWidget(thanks)

        link = QtWidgets.QLabel(
            f"<a href='{AUTHOR_WEBSITE}' "
            f"style='color:{COLOR_CYAN}; font-size:13pt; font-weight:700;'>"
            f"🌐 Acerca de mi creador </a>"
        )
        link.setAlignment(QtCore.Qt.AlignCenter)
        link.setOpenExternalLinks(True)
        link.setTextInteractionFlags(QtCore.Qt.TextBrowserInteraction)
        cv.addWidget(link)

        btn_web = QtWidgets.QPushButton("🚀  Visitar página web del proyecto")
        btn_web.setObjectName("webButton")
        btn_web.setMinimumHeight(44)
        btn_web.setCursor(QtCore.Qt.PointingHandCursor)
        btn_web.clicked.connect(
            lambda: QtGui.QDesktopServices.openUrl(
                QtCore.QUrl(PROJECT_WEBSITE))
        )
        cv.addWidget(btn_web)
        FX.pulse_glow(btn_web, COLOR_ACCENT, blur_min=10, blur_max=34)

        tech = QtWidgets.QLabel(
            f"<p style='color:{COLOR_TEXT_DIM}; font-size:9pt;'>"
            "Python · PyQt5 · NumPy · SciPy · PyWavelets · scikit-learn "
            "· XGBoost · LightGBM · Matplotlib</p>"
        )
        tech.setWordWrap(True)
        tech.setAlignment(QtCore.Qt.AlignCenter)
        cv.addWidget(tech)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1); row.addWidget(card); row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(4)

    # ------------------------------------------------------------------
    def showEvent(self, ev):
        super().showEvent(ev)
        self._timer.start(self.FPS_MS)
        FX.fade_in(self, FX.DURATION_MED)

    def hideEvent(self, ev):
        super().hideEvent(ev)
        self._timer.stop()      # costo cero mientras no se ve

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._make_stars()

    # ------------------------------------------------------------------
    def _make_stars(self):
        """~150 estrellas en coordenadas relativas (sobreviven resizes)."""
        rng = np.random.default_rng(42)
        self._stars = []
        for i in range(150):
            self._stars.append({
                "rx": float(rng.uniform(0, 1)),
                "ry": float(rng.uniform(0, 1)),
                "r":  float(rng.uniform(0.6, 1.9)),
                "a":  float(rng.uniform(70, 165)),
                "w":  float(rng.uniform(0.6, 2.4)),   # velocidad twinkle
                "ph": float(rng.uniform(0, 2 * np.pi)),
                "bright": bool(i < 12),               # 12 con destello
            })

    # ------------------------------------------------------------------
    def _update_comet(self, now: float, w: int, h: int):
        if self._comet is None and now >= self._next_comet:
            rng = self._rng
            # nace en el borde superior/izquierdo, cruza en diagonal
            if rng.uniform() < 0.5:
                x0, y0 = rng.uniform(0.05, 0.6) * w, -20.0
            else:
                x0, y0 = -20.0, rng.uniform(0.05, 0.5) * h
            speed = rng.uniform(260, 430)
            ang = rng.uniform(np.pi * 0.10, np.pi * 0.32)
            self._comet = {
                "x": x0, "y": y0,
                "vx": speed * np.cos(ang), "vy": speed * np.sin(ang),
                "tail": [], "size": rng.uniform(2.0, 3.4),
            }
        c = self._comet
        if c is None:
            return
        dt = self.FPS_MS / 1000.0
        c["x"] += c["vx"] * dt; c["y"] += c["vy"] * dt
        c["tail"].append((c["x"], c["y"]))
        if len(c["tail"]) > 26:
            c["tail"].pop(0)
        if c["x"] > w + 60 or c["y"] > h + 60:
            self._comet = None
            self._next_comet = now + float(self._rng.uniform(6.0, 13.0))

    def _draw_comet(self, p: QtGui.QPainter):
        c = self._comet
        if c is None or not c["tail"]:
            return
        n = len(c["tail"])
        for i, (tx, ty) in enumerate(c["tail"]):
            f = (i + 1) / n
            col = QtGui.QColor(COLOR_CYAN)
            col.setAlpha(int(150 * f * f))
            p.setBrush(col); p.setPen(QtCore.Qt.NoPen)
            r = c["size"] * (0.25 + 0.75 * f)
            p.drawEllipse(QtCore.QPointF(tx, ty), r, r)
        # cabeza brillante
        head = QtGui.QRadialGradient(c["x"], c["y"], c["size"] * 3)
        head.setColorAt(0.0, QtGui.QColor(255, 255, 255, 230))
        head.setColorAt(0.4, QtGui.QColor(COLOR_CYAN))
        head.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
        p.setBrush(QtGui.QBrush(head))
        p.drawEllipse(QtCore.QPointF(c["x"], c["y"]),
                      c["size"] * 3, c["size"] * 3)

    # ------------------------------------------------------------------
    def _draw_planet_sphere(self, p, x, y, r, color, sun_pos, alpha=255):
        """Esfera iluminada desde el sol: gradiente radial cuyo foco se
        desplaza hacia el sol → lado iluminado mirando al centro."""
        dx, dy = sun_pos[0] - x, sun_pos[1] - y
        d = max(1e-6, (dx * dx + dy * dy) ** 0.5)
        hx, hy = x + dx / d * r * 0.45, y + dy / d * r * 0.45
        g = QtGui.QRadialGradient(QtCore.QPointF(hx, hy), r * 2.0)
        base = QtGui.QColor(color)
        lit = base.lighter(165); lit.setAlpha(alpha)
        mid = base; mid.setAlpha(alpha)
        dark = base.darker(260); dark.setAlpha(alpha)
        g.setColorAt(0.0, lit)
        g.setColorAt(0.45, mid)
        g.setColorAt(1.0, dark)
        p.setBrush(QtGui.QBrush(g)); p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(QtCore.QPointF(x, y), r, r)

    def _draw_planet(self, p, idx, x, y, depth, sun_pos, now):
        name, color, _, r_base, _, extra = self.PLANETS[idx]
        # pseudo-3D: delante (depth>0) más grande/brillante
        scale = 1.0 + 0.22 * depth
        alpha = int(255 * (0.62 + 0.38 * (depth * 0.5 + 0.5)))
        r = r_base * scale

        if extra == "rings":     # mitad trasera del anillo
            pen = QtGui.QPen(QtGui.QColor(230, 210, 160,
                                          int(alpha * 0.75)))
            pen.setWidthF(2.0)
            p.setPen(pen); p.setBrush(QtCore.Qt.NoBrush)
            p.drawArc(QtCore.QRectF(x - 2.3 * r, y - 0.85 * r,
                                    4.6 * r, 1.7 * r),
                      0 * 16, 180 * 16)

        self._draw_planet_sphere(p, x, y, r, color, sun_pos, alpha)

        if extra == "bands":     # bandas de Júpiter
            p.save()
            path = QtGui.QPainterPath()
            path.addEllipse(QtCore.QPointF(x, y), r, r)
            p.setClipPath(path)
            band = QtGui.QColor("#b07c46"); band.setAlpha(int(alpha * .55))
            pen = QtGui.QPen(band); pen.setWidthF(r * 0.22)
            p.setPen(pen)
            for off in (-0.38, 0.05, 0.45):
                p.drawLine(QtCore.QPointF(x - r, y + off * r),
                           QtCore.QPointF(x + r, y + off * r))
            p.restore()

        if extra == "rings":     # mitad delantera del anillo
            pen = QtGui.QPen(QtGui.QColor(240, 222, 175, alpha))
            pen.setWidthF(2.4)
            p.setPen(pen); p.setBrush(QtCore.Qt.NoBrush)
            p.drawArc(QtCore.QRectF(x - 2.3 * r, y - 0.85 * r,
                                    4.6 * r, 1.7 * r),
                      180 * 16, 180 * 16)

        if extra == "moon":      # luna orbitando la Tierra
            mth = 2 * np.pi * (now / 2.6)
            mx = x + np.cos(mth) * r * 2.1
            my = y + np.sin(mth) * r * 2.1 * 0.5
            mr = r * 0.30
            if np.sin(mth) <= 0:    # detrás de la Tierra
                self._draw_planet_sphere(p, mx, my, mr, "#c9c9d4",
                                         sun_pos, int(alpha * 0.8))
            else:
                self._draw_planet_sphere(p, mx, my, mr, "#d8d8e2",
                                         sun_pos, alpha)

        # Nombre tenue bajo el planeta cuando está en la mitad frontal
        if depth >= 0.05:
            lbl = QtGui.QColor(COLOR_TEXT_DIM)
            lbl.setAlpha(int(60 + 70 * depth))
            p.setPen(lbl)
            f = QtGui.QFont("Segoe UI", 7)
            p.setFont(f)
            ring_off = 2.5 * r if extra == "rings" else 1.0 * r
            p.drawText(
                QtCore.QRectF(x - 60, y + ring_off + 4, 120, 14),
                QtCore.Qt.AlignHCenter | QtCore.Qt.AlignTop, name,
            )

    # ------------------------------------------------------------------
    def paintEvent(self, ev):
        w, h = self.width(), self.height()
        if w < 10 or h < 10:
            return
        now = time.monotonic() - self._t0
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        # 1) Espacio profundo
        bg = QtGui.QLinearGradient(0, 0, 0, h)
        bg.setColorAt(0.0, QtGui.QColor("#06070f"))
        bg.setColorAt(0.55, QtGui.QColor("#0a0d1c"))
        bg.setColorAt(1.0, QtGui.QColor("#10142a"))
        p.fillRect(self.rect(), QtGui.QBrush(bg))

        # 2) Nebulosas tenues
        for (rx, ry, rr, col, a) in (
            (0.18, 0.25, 0.55, COLOR_MAGENTA, 16),
            (0.82, 0.70, 0.60, COLOR_ACCENT, 14),
            (0.65, 0.15, 0.45, COLOR_CYAN, 10),
        ):
            neb = QtGui.QRadialGradient(rx * w, ry * h, rr * min(w, h))
            c0 = QtGui.QColor(col); c0.setAlpha(a)
            c1 = QtGui.QColor(col); c1.setAlpha(0)
            neb.setColorAt(0.0, c0); neb.setColorAt(1.0, c1)
            p.fillRect(self.rect(), QtGui.QBrush(neb))

        # 3) Estrellas titilantes
        if not self._stars:
            self._make_stars()
        p.setPen(QtCore.Qt.NoPen)
        for s in self._stars:
            tw = 0.55 + 0.45 * np.sin(s["w"] * now + s["ph"])
            a = int(s["a"] * tw)
            x, y = s["rx"] * w, s["ry"] * h
            col = QtGui.QColor(225, 232, 255, a)
            p.setBrush(col)
            p.drawEllipse(QtCore.QPointF(x, y), s["r"], s["r"])
            if s["bright"]:
                pen = QtGui.QPen(QtGui.QColor(225, 232, 255,
                                              int(a * 0.65)))
                pen.setWidthF(0.8)
                p.setPen(pen)
                L = s["r"] * (3.4 + 1.6 * tw)
                p.drawLine(QtCore.QPointF(x - L, y), QtCore.QPointF(x + L, y))
                p.drawLine(QtCore.QPointF(x, y - L), QtCore.QPointF(x, y + L))
                p.setPen(QtCore.Qt.NoPen)

        # 4) Cometa
        self._update_comet(time.monotonic(), w, h)
        self._draw_comet(p)

        # 5) Sistema solar — centro elevado para que el sol no quede
        #    tapado por la tarjeta de contenido
        cx, cy = w * 0.5, h * 0.40
        S = min(w * 0.55, h * 0.95)
        sun_pos = (cx, cy)

        # Órbitas
        pen = QtGui.QPen(QtGui.QColor(122, 162, 247, 26))
        pen.setWidthF(1.0)
        p.setPen(pen); p.setBrush(QtCore.Qt.NoBrush)
        for (_, _, r_orb, _, _, _) in self.PLANETS:
            a_ax = r_orb * S; b_ax = a_ax * self.TILT
            p.drawEllipse(QtCore.QPointF(cx, cy), a_ax, b_ax)

        # Posiciones + estelas
        positions = []
        for i, (name, color, r_orb, r_px, period, extra) in \
                enumerate(self.PLANETS):
            th = 2 * np.pi * (now / period) + self._phases[i]
            a_ax = r_orb * S; b_ax = a_ax * self.TILT
            x = cx + a_ax * np.cos(th)
            y = cy + b_ax * np.sin(th)
            depth = float(np.sin(th))
            positions.append((i, x, y, depth, color))
            tr = self._trails[i]
            tr.append((x, y, depth))
            if len(tr) > self.TRAIL_LEN:
                tr.pop(0)

        def draw_trail(i, color):
            tr = self._trails[i]
            n = len(tr)
            p.setPen(QtCore.Qt.NoPen)
            for j, (tx, ty, td) in enumerate(tr):
                f = (j + 1) / n
                col = QtGui.QColor(color)
                col.setAlpha(int(70 * f * (0.55 + 0.45 * (td * .5 + .5))))
                p.setBrush(col)
                rr = 1.0 + 1.6 * f
                p.drawEllipse(QtCore.QPointF(tx, ty), rr, rr)

        # 5a) planetas DETRÁS del sol
        for (i, x, y, depth, color) in positions:
            if depth < 0:
                draw_trail(i, color)
                self._draw_planet(p, i, x, y, depth, sun_pos, now)

        # 5b) SOL (halo → corona → núcleo, con pulso)
        rs = S * 0.085 * (1.0 + 0.05 * np.sin(1.25 * now))
        halo = QtGui.QRadialGradient(cx, cy, rs * 4.6)
        halo.setColorAt(0.0, QtGui.QColor(255, 200, 90, 70))
        halo.setColorAt(0.45, QtGui.QColor(255, 160, 60, 26))
        halo.setColorAt(1.0, QtGui.QColor(255, 140, 40, 0))
        p.setBrush(QtGui.QBrush(halo)); p.setPen(QtCore.Qt.NoPen)
        p.drawEllipse(QtCore.QPointF(cx, cy), rs * 4.6, rs * 4.6)
        corona = QtGui.QRadialGradient(cx, cy, rs * 2.1)
        corona.setColorAt(0.0, QtGui.QColor(255, 236, 170, 200))
        corona.setColorAt(0.65, QtGui.QColor(255, 180, 80, 120))
        corona.setColorAt(1.0, QtGui.QColor(255, 150, 50, 0))
        p.setBrush(QtGui.QBrush(corona))
        p.drawEllipse(QtCore.QPointF(cx, cy), rs * 2.1, rs * 2.1)
        core = QtGui.QRadialGradient(cx - rs * .25, cy - rs * .25, rs * 1.25)
        core.setColorAt(0.0, QtGui.QColor("#fff7da"))
        core.setColorAt(0.55, QtGui.QColor("#ffd566"))
        core.setColorAt(1.0, QtGui.QColor("#ff9c3a"))
        p.setBrush(QtGui.QBrush(core))
        p.drawEllipse(QtCore.QPointF(cx, cy), rs, rs)

        # 5c) planetas DELANTE del sol
        for (i, x, y, depth, color) in positions:
            if depth >= 0:
                draw_trail(i, color)
                self._draw_planet(p, i, x, y, depth, sun_pos, now)

        p.end()
        # Los hijos (botón Volver + tarjeta glass) se pintan después →
        # siempre quedan ENCIMA del cosmos.


class AboutPage(QtWidgets.QWidget):
    """
    Página «Acerca de» a PANTALLA COMPLETA (v4.6) — reemplaza al antiguo
    diálogo de ayuda emergente. Es la página 2 del QStackedWidget
    central. Se entra con el botón «ℹ» del header y se vuelve con
    «← Volver» o Esc.

    El fondo es una RED NEURONAL 3D animada pintada con QPainter:
      • Arquitectura 3-6-8-6-4-2 (entrada → 4 ocultas → salida),
        con los nodos de cada capa repartidos en una rejilla vertical y
        una coordenada z propia para dar volumen.
      • Toda la nube de neuronas rota lentamente en los ejes Y y X;
        cada nodo se proyecta en perspectiva (los de atrás se ven más
        pequeños y tenues → profundidad real).
      • Las conexiones entre capas tienen un peso pseudo-aleatorio fijo
        que define su color (cian = positivo, magenta = negativo) y
        grosor. Se dibujan ordenadas por profundidad (painter's
        algorithm) para que el 3D se lea bien.
      • PULSOS DE ACTIVACIÓN: continuamente salen «forward passes» —
        ondas de luz que viajan capa a capa por las aristas, iluminando
        las neuronas al llegar (un nodo activado crece y brilla). Es la
        metáfora visual de la inferencia de la red.
      • Campo de partículas de fondo + viñeta radial para profundidad.
    ~30 fps SOLO mientras la página está visible (timer en
    showEvent/hideEvent → costo cero en uso normal).
    """

    back_requested = QtCore.pyqtSignal()

    LAYERS = [3, 6, 8, 6, 4, 2]   # arquitectura mostrada
    FPS_MS = 33                    # ~30 fps
    PULSE_PERIOD = 1.7             # s entre forward-passes nuevos
    PULSE_SPEED = 2.6              # capas por segundo

    def __init__(self, parent=None):
        super().__init__(parent)
        self._t0 = time.monotonic()
        self._rng = np.random.default_rng(11)
        self._build_net()
        self._pulses: List[Dict[str, Any]] = []
        self._next_pulse = 0.0
        self._particles = []
        self._timer = QtCore.QTimer(self)
        self._timer.timeout.connect(self.update)
        self._build_overlay_ui()

        esc = QtWidgets.QShortcut(QtGui.QKeySequence("Esc"), self)
        esc.setContext(QtCore.Qt.WidgetWithChildrenShortcut)
        esc.activated.connect(self.back_requested.emit)

    # ------------------------------------------------------------------
    def _build_net(self):
        """Genera posiciones 3D de cada neurona y los pesos de cada
        conexión (una sola vez; el movimiento es solo rotación)."""
        rng = self._rng
        self._nodes: List[List[Dict[str, Any]]] = []
        n_layers = len(self.LAYERS)
        for li, n in enumerate(self.LAYERS):
            # x: avanza por capa, centrado en 0 — rango amplio para que
            # la red se extienda horizontalmente por toda la pantalla.
            x = (li / (n_layers - 1) - 0.5) * 3.0      # −1.5 … +1.5
            layer = []
            for j in range(n):
                y = (j / max(1, n - 1) - 0.5) * 2.1     # rejilla vertical
                # z: dispersión para dar volumen 3D a la capa
                z = (rng.uniform(-0.6, 0.6)
                     if n > 1 else 0.0)
                layer.append({
                    "base": np.array([x, y, z], dtype=float),
                    "act": 0.0,           # nivel de activación (0..1)
                })
            self._nodes.append(layer)
        # Pesos: weights[li][a][b] conecta nodo a de capa li con b de li+1
        self._weights: List[np.ndarray] = []
        for li in range(n_layers - 1):
            na, nb = self.LAYERS[li], self.LAYERS[li + 1]
            W = rng.normal(0, 1, size=(na, nb))
            self._weights.append(W)

    # ------------------------------------------------------------------
    def _build_overlay_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(18, 14, 18, 18)

        top = QtWidgets.QHBoxLayout()
        self.btn_back = QtWidgets.QPushButton("←  Volver al programa")
        self.btn_back.setCursor(QtCore.Qt.PointingHandCursor)
        self.btn_back.setMinimumHeight(38)
        self.btn_back.setStyleSheet(f"""
            QPushButton {{
                background: rgba(13, 15, 26, 0.72); color: {COLOR_TEXT};
                border: 1px solid rgba(122, 162, 247, 0.45);
                border-radius: 9px; padding: 7px 16px; font-weight: 600;
            }}
            QPushButton:hover {{
                color: {COLOR_CYAN}; border-color: {COLOR_CYAN};
                background: rgba(18, 22, 38, 0.85);
            }}
        """)
        self.btn_back.clicked.connect(self.back_requested.emit)
        top.addWidget(self.btn_back, 0, QtCore.Qt.AlignLeft)
        top.addStretch(1)
        hint = QtWidgets.QLabel("Esc para volver")
        hint.setStyleSheet("color: rgba(169,177,214,0.55); font-size: 8pt;"
                           "background: transparent;")
        top.addWidget(hint, 0, QtCore.Qt.AlignRight)
        outer.addLayout(top)

        outer.addStretch(1)

        # Tarjeta glass con información del programa
        card = QtWidgets.QFrame()
        card.setObjectName("aboutCard")
        card.setMaximumWidth(760)
        card.setStyleSheet(f"""
            QFrame#aboutCard {{
                background: rgba(8, 10, 20, 0.66);
                border: 1px solid rgba(122, 162, 247, 0.34);
                border-radius: 18px;
            }}
            QLabel {{ background: transparent; color: {COLOR_TEXT}; }}
        """)
        cv = QtWidgets.QVBoxLayout(card)
        cv.setContentsMargins(40, 30, 40, 30); cv.setSpacing(14)

        title = QtWidgets.QLabel("🧠  ¿Qué es HydroAnalyzer?")
        title.setStyleSheet(
            f"font-size: 22pt; font-weight: 800; color: {COLOR_ACCENT};"
            f"background: transparent;"
        )
        title.setAlignment(QtCore.Qt.AlignCenter)
        cv.addWidget(title)

        intro = QtWidgets.QLabel(
            "<p style='font-size:11pt; line-height:148%;'>"
            "<b>HydroAnalyzer</b> es una plataforma de escritorio que "
            "detecta <b>conexiones clandestinas (bypass)</b> en redes de "
            "tuberías a partir del análisis del <b>golpe de ariete</b> "
            "(transiente de presión). Combina procesamiento de señales "
            "con modelos de <b>machine learning</b> entrenados para "
            "distinguir una tubería íntegra de una intervenida.</p>"
        )
        intro.setWordWrap(True)
        cv.addWidget(intro)

        # Grid de "cómo funciona" en 4 pasos
        steps = QtWidgets.QLabel(
            "<table width='100%' cellspacing='0' cellpadding='7'>"
            "<tr>"
            "<td width='50%'><span style='color:#7aa2f7; font-weight:700;'>"
            "1 · Simulación &amp; captura</span><br>"
            "<span style='color:#a9b1d6;'>Genera transientes sintéticos o "
            "carga señales reales de presión (CSV).</span></td>"
            "<td width='50%'><span style='color:#7dcfff; font-weight:700;'>"
            "2 · Filtrado &amp; features</span><br>"
            "<span style='color:#a9b1d6;'>Limpia la señal y extrae 17 "
            "descriptores: energía por banda, wavelets, picos, etc.</span>"
            "</td></tr>"
            "<tr>"
            "<td><span style='color:#bb9af7; font-weight:700;'>"
            "3 · Entrenamiento</span><br>"
            "<span style='color:#a9b1d6;'>RF, SVM, XGBoost y LightGBM con "
            "aumentación de datos y calibración de probabilidades.</span></td>"
            "<td><span style='color:#9ece6a; font-weight:700;'>"
            "4 · Validación &amp; análisis</span><br>"
            "<span style='color:#a9b1d6;'>Validación cruzada, ranking de "
            "modelos, detección de outliers y predicción en vivo.</span></td>"
            "</tr></table>"
        )
        steps.setWordWrap(True)
        cv.addWidget(steps)

        models = QtWidgets.QLabel(
            f"<p style='color:{COLOR_TEXT_DIM}; font-size:9.5pt; "
            f"text-align:center;'>"
            "los modelos en producción son ensembles de árboles y SVM, elegidos por su "
            "elegidos por su robustez e interpretabilidad sobre datasets pequeños.</p>"
        )
        models.setWordWrap(True)
        models.setAlignment(QtCore.Qt.AlignCenter)
        cv.addWidget(models)

        row = QtWidgets.QHBoxLayout()
        row.addStretch(1); row.addWidget(card); row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(2)

    # ------------------------------------------------------------------
    def showEvent(self, ev):
        super().showEvent(ev)
        self._timer.start(self.FPS_MS)
        FX.fade_in(self, FX.DURATION_MED)

    def hideEvent(self, ev):
        super().hideEvent(ev)
        self._timer.stop()

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self._make_particles()

    def _make_particles(self):
        rng = np.random.default_rng(5)
        self._particles = [{
            "rx": float(rng.uniform(0, 1)), "ry": float(rng.uniform(0, 1)),
            "r": float(rng.uniform(0.5, 1.6)), "a": float(rng.uniform(30, 90)),
            "w": float(rng.uniform(0.4, 1.4)), "ph": float(rng.uniform(0, 6.28)),
        } for _ in range(70)]

    # ------------------------------------------------------------------
    def _project(self, v3, ay, ax, w, h, scale, scale_x=None):
        """Rota un punto 3D (ejes Y y X) y lo proyecta en perspectiva.
        Devuelve (x2d, y2d, depth01, f). scale_x permite estirar el eje
        horizontal (red más ancha que alta)."""
        if scale_x is None:
            scale_x = scale
        x, y, z = v3
        # rotación Y
        cy, sy = np.cos(ay), np.sin(ay)
        x, z = x * cy + z * sy, -x * sy + z * cy
        # rotación X
        cx, sx = np.cos(ax), np.sin(ax)
        y, z = y * cx - z * sx, y * sx + z * cx
        # perspectiva: cámara a distancia d sobre el eje z
        d = 3.4
        f = d / (d - z)            # factor de perspectiva
        sx2 = w * 0.5 + x * scale_x * f
        sy2 = h * 0.42 + y * scale * f
        depth01 = float(np.clip((z + 1.5) / 3.0, 0.0, 1.0))
        return sx2, sy2, depth01, f

    # ------------------------------------------------------------------
    def _update_pulses(self, now):
        # lanzar un nuevo forward-pass periódicamente
        if now >= self._next_pulse:
            self._pulses.append({"t0": now})
            self._next_pulse = now + self.PULSE_PERIOD
        # limpiar activaciones
        for layer in self._nodes:
            for nd in layer:
                nd["act"] *= 0.85       # decaimiento suave
        alive = []
        n_layers = len(self.LAYERS)
        for pulse in self._pulses:
            prog = (now - pulse["t0"]) * self.PULSE_SPEED  # en "capas"
            seg = int(np.floor(prog))                       # capa origen
            if seg >= n_layers - 1:
                # llegó a la salida → encender capa final y morir
                for nd in self._nodes[-1]:
                    nd["act"] = 1.0
                continue
            # iluminar la capa que el frente está alcanzando
            frac = prog - seg
            if frac < 0.5:
                for nd in self._nodes[seg]:
                    nd["act"] = max(nd["act"], 1.0)
            pulse["seg"] = seg
            pulse["frac"] = frac
            alive.append(pulse)
        self._pulses = alive

    # ------------------------------------------------------------------
    def paintEvent(self, ev):
        w, h = self.width(), self.height()
        if w < 10 or h < 10:
            return
        now = time.monotonic() - self._t0
        p = QtGui.QPainter(self)
        p.setRenderHint(QtGui.QPainter.Antialiasing)

        # 1) Fondo profundo + viñeta
        bg = QtGui.QLinearGradient(0, 0, 0, h)
        bg.setColorAt(0.0, QtGui.QColor("#080a14"))
        bg.setColorAt(1.0, QtGui.QColor("#0e1124"))
        p.fillRect(self.rect(), QtGui.QBrush(bg))
        vig = QtGui.QRadialGradient(w * 0.5, h * 0.42, max(w, h) * 0.7)
        vig.setColorAt(0.0, QtGui.QColor(122, 162, 247, 18))
        vig.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
        p.fillRect(self.rect(), QtGui.QBrush(vig))

        # 2) Partículas
        if not self._particles:
            self._make_particles()
        p.setPen(QtCore.Qt.NoPen)
        for s in self._particles:
            a = int(s["a"] * (0.5 + 0.5 * np.sin(s["w"] * now + s["ph"])))
            p.setBrush(QtGui.QColor(150, 180, 255, a))
            p.drawEllipse(QtCore.QPointF(s["rx"] * w, s["ry"] * h),
                          s["r"], s["r"])

        # 3) Red neuronal
        self._update_pulses(now)
        ay = now * 0.32                     # giro continuo en Y
        ax = 0.30 * np.sin(now * 0.21)      # leve cabeceo en X
        # escala anisotrópica: más ancho que alto para llenar la pantalla
        scale = min(w, h) * 0.46
        scale_x = scale * 1.35

        # proyectar todos los nodos
        proj: List[List[Tuple[float, float, float, float]]] = []
        for layer in self._nodes:
            pl = [self._project(nd["base"], ay, ax, w, h, scale, scale_x)
                  for nd in layer]
            proj.append(pl)

        # 3a) conexiones, ordenadas por profundidad media (lejos→cerca)
        edges = []
        for li in range(len(self.LAYERS) - 1):
            W = self._weights[li]
            for a in range(self.LAYERS[li]):
                xa, ya2, da, _ = proj[li][a]
                for b in range(self.LAYERS[li + 1]):
                    xb, yb2, db, _ = proj[li + 1][b]
                    edges.append((
                        0.5 * (da + db), li, a, b,
                        xa, ya2, xb, yb2, W[a, b],
                    ))
        edges.sort(key=lambda e: e[0])     # dibujar primero los lejanos
        for (depth, li, a, b, xa, ya2, xb, yb2, wgt) in edges:
            # color por signo del peso, alpha por profundidad
            pos = wgt >= 0
            base = QtGui.QColor(COLOR_CYAN if pos else COLOR_MAGENTA)
            a_edge = int((22 + 40 * depth) * min(1.0, 0.4 + abs(wgt)))
            base.setAlpha(a_edge)
            pen = QtGui.QPen(base)
            pen.setWidthF(max(0.4, abs(wgt) * 1.5 * (0.5 + depth)))
            p.setPen(pen)
            p.drawLine(QtCore.QPointF(xa, ya2), QtCore.QPointF(xb, yb2))

        # 3b) pulsos viajando por las aristas (forward-pass)
        for pulse in self._pulses:
            seg = pulse.get("seg")
            if seg is None:
                continue
            frac = pulse["frac"]
            for a in range(self.LAYERS[seg]):
                xa, ya2, _, _ = proj[seg][a]
                for b in range(self.LAYERS[seg + 1]):
                    xb, yb2, _, _ = proj[seg + 1][b]
                    px = xa + (xb - xa) * frac
                    py = ya2 + (yb2 - ya2) * frac
                    glow = QtGui.QRadialGradient(px, py, 7)
                    glow.setColorAt(0.0, QtGui.QColor(255, 255, 255, 210))
                    glow.setColorAt(0.4, QtGui.QColor(COLOR_CYAN))
                    glow.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
                    p.setBrush(QtGui.QBrush(glow)); p.setPen(QtCore.Qt.NoPen)
                    p.drawEllipse(QtCore.QPointF(px, py), 7, 7)

        # 3c) nodos, ordenados por profundidad
        node_list = []
        for li, layer in enumerate(self._nodes):
            for j, nd in enumerate(layer):
                x2, y2, depth, f = proj[li][j]
                node_list.append((depth, x2, y2, f, nd["act"], li))
        node_list.sort(key=lambda n: n[0])
        for (depth, x2, y2, f, act, li) in node_list:
            r = (6.5 + 4.0 * depth) * f * (1.0 + 0.5 * act)
            # color base de la capa
            if li == 0:
                col = QtGui.QColor(COLOR_SUCCESS)        # entrada
            elif li == len(self.LAYERS) - 1:
                col = QtGui.QColor("#e0af68")            # salida
            else:
                col = QtGui.QColor(COLOR_ACCENT)         # ocultas
            # halo de activación
            if act > 0.05:
                halo = QtGui.QRadialGradient(x2, y2, r * 3.2)
                hc = QtGui.QColor(255, 255, 255)
                hc.setAlpha(int(150 * act))
                halo.setColorAt(0.0, hc)
                halo.setColorAt(1.0, QtGui.QColor(0, 0, 0, 0))
                p.setBrush(QtGui.QBrush(halo)); p.setPen(QtCore.Qt.NoPen)
                p.drawEllipse(QtCore.QPointF(x2, y2), r * 3.2, r * 3.2)
            # esfera del nodo (gradiente para dar volumen)
            g = QtGui.QRadialGradient(x2 - r * 0.3, y2 - r * 0.3, r * 1.8)
            lit = col.lighter(int(150 + 80 * act))
            lit.setAlpha(int(120 + 135 * depth))
            dk = col.darker(240); dk.setAlpha(int(120 + 135 * depth))
            g.setColorAt(0.0, lit); g.setColorAt(1.0, dk)
            p.setBrush(QtGui.QBrush(g))
            pen = QtGui.QPen(col.lighter(160))
            pen.setWidthF(0.8); p.setPen(pen)
            p.drawEllipse(QtCore.QPointF(x2, y2), r, r)

        p.end()
        # tarjeta + botón Volver se pintan encima (son hijos)


class AnimatedHeaderFrame(QtWidgets.QFrame):
    """
    Cabecera con fondo animado de símbolos (v4.3) — reemplaza al antiguo
    overlay de ventana completa de v4.2.

    Diferencia clave: los símbolos NO van en un widget superpuesto, sino
    que se pintan DENTRO del ``paintEvent`` de este frame, en este orden:
        1) el fondo del stylesheet (gradiente + borde + radio), vía
           QStyleOption + PE_Widget — el patrón oficial de Qt para
           subclases custom que respetan su stylesheet;
        2) los símbolos, recortados al rectángulo redondeado;
        3) los hijos (título, subtítulo, badge de modelo, botones) se
           pintan DESPUÉS por el propio Qt → quedan SIEMPRE encima.
    Resultado: un background real, detrás de las letras, confinado a la
    banda superior.

    Movimiento: deriva horizontal lenta hacia la izquierda con un leve
    vaivén vertical sinusoidal; al salir por la izquierda, el símbolo
    renace por la derecha con texto/tamaño/color nuevos.

    Desactivable con la variable de entorno HYDRO_NO_BG=1 (igual que el
    splash con HYDRO_NO_SPLASH=1).
    """

    SYMBOLS = [
        # Matemáticas
        "∫", "∂", "Σ", "λ", "μ", "π", "Ω", "θ", "∈", "≈", "√", "∞",
        "∇", "δ", "eˣ", "f(x)", "lim", "dy/dx", "∮", "Δ", "α", "β",
        "x→0", "ℝ", "∝", "≠", "±",
        # Ingeniería / física / hidráulica
        "UART", "PID", "DSP", "PLC", "PWM", "ΔP", "H₂O", "EEG",
        "GPU", "ASM", "0x1F", "404", "I²C", "SPI", "ADC", "kPa",
        "f₀", "τ", "Re", "Q=VA",
        # IA / programación
        "Python", "Arduino", "C++", "C#", "ML", "AI", "RF", "SVM",
        "XGB", "if(x)", "while", "import", "def", "λ=c/f", "np.fft",
        "train()", "y=σ(x)", "// TODO",
    ]
    # Paleta de tintes (se sortea por símbolo)
    TINTS = None   # se rellena en __init__ (necesita los COLOR_*)

    N_ITEMS = 16   # banda angosta → menos símbolos que el overlay viejo
    FPS_MS  = 40   # ~25 fps

    def __init__(self, parent=None):
        super().__init__(parent)
        # El stylesheet global estiliza QFrame#header: conservamos el
        # objectName para heredar el gradiente, borde y radio.
        self.setObjectName("header")

        if AnimatedHeaderFrame.TINTS is None:
            AnimatedHeaderFrame.TINTS = [
                COLOR_ACCENT, COLOR_CYAN, COLOR_MAGENTA,
                COLOR_SUCCESS, COLOR_TEXT_DIM, "#e0af68",
            ]

        self._rng = np.random.default_rng(99)
        self._items: List[Dict[str, Any]] = []
        self._t_last = time.monotonic()
        self._enabled = os.environ.get("HYDRO_NO_BG", "0") != "1"

        if self._enabled:
            self._timer = QtCore.QTimer(self)
            self._timer.timeout.connect(self._tick)
            self._timer.start(self.FPS_MS)

    # ------------------------------------------------------------------
    def _new_item(self, at_right: bool = False) -> Dict[str, Any]:
        """Crea un símbolo. at_right=True lo hace nacer fuera del borde
        derecho (para el reciclaje); False lo reparte por toda la banda
        (población inicial)."""
        w = max(1, self.width()); h = max(1, self.height())
        rng = self._rng
        text = str(rng.choice(self.SYMBOLS))
        size = int(rng.integers(8, 15))
        font = QtGui.QFont("Consolas", size)
        font.setStyleHint(QtGui.QFont.Monospace)
        fm = QtGui.QFontMetricsF(font)
        return {
            "text":  text,
            "font":  font,
            "tw":    float(fm.horizontalAdvance(text)),
            "x":     float(w + rng.uniform(10, 160)) if at_right
                     else float(rng.uniform(0, w)),
            # baseline dentro de la banda, con margen para el bob
            "y":     float(rng.uniform(16, max(18, h - 8))),
            "vx":    float(rng.uniform(8.0, 26.0)),    # px/s hacia la izq.
            "amp":   float(rng.uniform(1.5, 4.5)),     # bob vertical
            "omega": float(rng.uniform(0.3, 0.9)),     # rad/s del bob
            "phase": float(rng.uniform(0, 2 * np.pi)),
            "alpha": int(rng.integers(20, 42)),        # ≈8–16 % — sutil
            "color": str(rng.choice(self.TINTS)),
            "t0":    time.monotonic(),
        }

    # ------------------------------------------------------------------
    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        # Población perezosa: el primer resize con ancho real define la
        # distribución inicial (en __init__ el frame aún mide 0×0).
        if self._enabled and not self._items and self.width() > 50:
            self._items = [self._new_item(at_right=False)
                           for _ in range(self.N_ITEMS)]

    # ------------------------------------------------------------------
    def _tick(self):
        if not self._enabled or not self.isVisible():
            return
        win = self.window()
        if win is not None and win.isMinimized():
            return    # ventana minimizada → CPU ~0
        now = time.monotonic()
        dt = min(0.1, now - self._t_last)   # clamp por si hubo freeze
        self._t_last = now
        if not self._items and self.width() > 50:
            self._items = [self._new_item(at_right=False)
                           for _ in range(self.N_ITEMS)]
        for it in self._items:
            it["x"] -= it["vx"] * dt
            if it["x"] + it["tw"] < -10:    # salió por la izquierda
                it.update(self._new_item(at_right=True))
        # La banda es pequeña (~ancho × 74 px): repintarla entera cada
        # tick es barato y más simple que regiones sucias.
        self.update()

    # ------------------------------------------------------------------
    def paintEvent(self, ev):
        p = QtGui.QPainter(self)
        # 1) Fondo del stylesheet (gradiente + borde + radio). Patrón
        #    documentado de Qt para widgets custom con stylesheet.
        opt = QtWidgets.QStyleOption()
        opt.initFrom(self)
        self.style().drawPrimitive(QtWidgets.QStyle.PE_Widget, opt, p, self)

        # 2) Símbolos, recortados al rounded-rect para respetar esquinas.
        if self._enabled and self._items:
            p.setRenderHint(QtGui.QPainter.Antialiasing)
            p.setRenderHint(QtGui.QPainter.TextAntialiasing)
            path = QtGui.QPainterPath()
            path.addRoundedRect(
                QtCore.QRectF(self.rect()).adjusted(1, 1, -1, -1), 12, 12
            )
            p.setClipPath(path)
            now = time.monotonic()
            for it in self._items:
                bob = it["amp"] * np.sin(
                    it["omega"] * (now - it["t0"]) + it["phase"]
                )
                c = QtGui.QColor(it["color"]); c.setAlpha(it["alpha"])
                p.setPen(c)
                p.setFont(it["font"])
                p.drawText(QtCore.QPointF(it["x"], it["y"] + bob),
                           it["text"])
        p.end()
        # 3) Los hijos (título, badge, botones) los pinta Qt después de
        #    este método → siempre quedan ENCIMA de los símbolos.


# ============================================================================
# 2. SEÑAL Y CARACTERÍSTICAS
# ============================================================================

@dataclass
class TransientParams:
    duration: float = 5.0
    fs: int = 2000
    p0: float = 2.5
    t0: float = 0.5
    A: float = 0.6
    f0: float = 25.0
    tau: float = 0.4
    noise_std: float = 0.01
    bypass: bool = False
    seed: Optional[int] = None


def generate_transient(params: TransientParams) -> Tuple[np.ndarray, np.ndarray]:
    if params.seed is not None:
        np.random.seed(params.seed)
    t = np.arange(0, params.duration, 1.0 / params.fs)
    y = np.ones_like(t) * params.p0

    A, tau, f0 = params.A, params.tau, params.f0
    if params.bypass:
        A, tau, f0 = A * 0.6, tau * 0.6, f0 * 1.2

    idx = t >= params.t0
    env = np.exp(-(t - params.t0) / tau) * idx
    y += A * env * np.sin(2 * np.pi * f0 * (t - params.t0))
    y += 0.02 * env * np.sin(2 * np.pi * 200 * (t - params.t0))
    y += np.random.normal(0, params.noise_std, size=y.shape)
    return t, y


def extract_features(t: np.ndarray, p: np.ndarray, fs: int) -> Dict[str, float]:
    features: Dict[str, float] = {}

    baseline_window = max(1, int(0.1 * len(p)))
    p0 = float(np.median(p[:baseline_window]))
    features["baseline"] = p0

    peak_idx = int(np.argmax(p))
    features["peak_amp"] = float(p[peak_idx] - p0)
    features["t_peak"]   = float(t[peak_idx]) if peak_idx < len(t) else 0.0
    features["rms"]      = float(np.sqrt(np.mean((p - p0) ** 2)))
    features["crest"]    = float(np.max(np.abs(p - p0)) / (features["rms"] + 1e-9))

    try:
        mask = (t >= t[peak_idx]) & (t <= t[peak_idx] + 1.0)
        if np.sum(mask) > 10:
            env = np.abs(p[mask] - p0)
            env[env <= 1e-6] = 1e-6

            def expo(x, a, tau):
                return a * np.exp(-(x - t[peak_idx]) / tau)

            popt, _ = curve_fit(expo, t[mask], env, p0=[env[0], 0.3], maxfev=5000)
            features["decay_tau"] = float(max(popt[1], 1e-3))
        else:
            features["decay_tau"] = 0.0
    except Exception:
        features["decay_tau"] = 0.0

    N = len(p)
    try:
        yf = np.abs(rfft(p - p0))
        xf = rfftfreq(N, 1.0 / fs)
    except Exception:
        yf = np.array([0.0])
        xf = np.array([0.0])

    features["energy_total"] = float(np.sum(yf ** 2))
    for a, b in FREQ_BANDS:
        band_mask = (xf >= a) & (xf < b)
        features[f"energy_band_{a}_{b}"] = float(np.sum(yf[band_mask] ** 2))
    features["dom_freq"] = float(xf[np.argmax(yf)]) if np.sum(yf) > 0 else 0.0

    try:
        coeffs = pywt.wavedec(p - p0, WAVELET_NAME, level=WAVELET_LEVEL)
        for i, c in enumerate(coeffs):
            features[f"wavelet_E_{i}"] = float(np.sum(np.asarray(c) ** 2))
    except Exception:
        for i in range(WAVELET_LEVEL + 1):
            features[f"wavelet_E_{i}"] = 0.0

    return features


def features_to_vector(feats: Dict[str, float],
                       feature_names: Optional[List[str]] = None
                       ) -> Tuple[np.ndarray, List[str]]:
    names = feature_names if feature_names is not None else sorted(feats.keys())
    vec = np.array([feats.get(k, 0.0) for k in names], dtype=float)
    return vec, names


# ============================================================================
# 3. DATA AUGMENTATION
# ============================================================================

def add_noise(p, noise_std):
    return p + np.random.normal(0, noise_std, size=p.shape)


def scale_amplitude(p, factor):
    return p * factor


def time_shift(t, p, shift_s):
    if len(t) <= 1:
        return t, p
    dt = float(np.mean(np.diff(t)))
    if dt == 0:
        return t, p
    n = int(round(shift_s / dt))
    p2 = np.roll(p, n)
    if n > 0:
        p2[:n] = p[0]
    elif n < 0:
        p2[n:] = p[-1]
    return t.copy(), p2


def time_stretch(t, p, stretch_factor):
    if stretch_factor == 1.0 or len(t) < 2:
        return t.copy(), p.copy()
    N = len(p)
    new_len = max(2, int(N * stretch_factor))
    new_t = np.linspace(t[0], t[-1], new_len)
    new_p = np.interp(new_t, t, p)
    res_t = np.linspace(new_t[0], new_t[-1], N)
    res_p = np.interp(res_t, new_t, new_p)
    return res_t, res_p


def augment_single_signal(t, p, n_aug: int = 5,
                          noise_range=(0.002, 0.02),
                          amp_range=(0.9, 1.12),
                          shift_seconds=(-0.02, 0.02),
                          stretch_range=(0.95, 1.05)):
    augmented = [(t.copy(), p.copy())]  # original
    for _ in range(n_aug):
        t1, p1 = t.copy(), p.copy()
        if np.random.rand() < 0.9:
            p1 = add_noise(p1, np.random.uniform(*noise_range))
        if np.random.rand() < 0.7:
            p1 = scale_amplitude(p1, np.random.uniform(*amp_range))
        if np.random.rand() < 0.6:
            t1, p1 = time_shift(t1, p1, np.random.uniform(*shift_seconds))
        if np.random.rand() < 0.5:
            t1, p1 = time_stretch(t1, p1, np.random.uniform(*stretch_range))
        augmented.append((t1, p1))
    return augmented


# ============================================================================
# 3.5 FILTRADO DE SEÑAL  (anti-spike + pasa-bajos)
# ============================================================================

@dataclass
class ManualInterval:
    """
    Define un intervalo de tiempo donde eliminar manualmente picos que
    superan/no llegan a un cierto umbral de presión. Útil para suprimir
    picos puntuales que el usuario identifica visualmente en la gráfica
    y que escapan a los filtros automáticos.
    """
    t_start: float = 0.0          # inicio del intervalo (s)
    t_end: float = 1.0            # fin del intervalo (s)
    threshold: float = 5.0        # umbral de presión (bar)
    mode: str = ">"               # ">" elimina x>thr, "<" elimina x<thr
    enabled: bool = True


@dataclass
class FilterConfig:
    """Configuración del pipeline de filtrado.

    En v3.6 TODOS los sub-filtros y el master están APAGADOS por defecto.
    El usuario debe activar explícitamente cada etapa que quiera usar.
    """
    enabled: bool = False          # master (apagado por defecto)
    # 1) Diferencia con vecinos (anti-spike quirúrgico de 1 muestra)
    neighbor_enabled: bool = False
    neighbor_threshold_sigmas: float = 4.0
    neighbor_max_passes: int = 2
    neighbor_agree_ratio: float = 0.5
    # 2) Hampel (mediana móvil + MAD)
    hampel_enabled: bool = False
    hampel_window: int = 7         # nº de muestras (impar)
    hampel_n_sigmas: float = 3.0
    # 3) Envolvente IQR (anti-spike de racimos densos)
    iqr_enabled: bool = False
    iqr_window: int = 31           # ventana del rolling-quantile
    iqr_k: float = 3.0             # extensión sobre IQR como umbral
    iqr_max_passes: int = 3
    iqr_protect_transients: bool = True
    iqr_protect_window: int = 201  # ventana para detectar transientes reales
    # 4) Filtro consciente de duración (distingue spikes de transientes reales)
    duration_enabled: bool = False
    duration_baseline_window: int = 401   # ventana de la base (muestras)
    duration_k_sigmas: float = 3.5        # umbral en sigmas robustos
    duration_min_transient_s: float = 0.05  # duración mínima de un transiente real
    duration_max_passes: int = 3
    # 5) Eliminación manual por intervalo
    manual_enabled: bool = False
    manual_intervals: List[ManualInterval] = field(default_factory=list)
    # 6) Pasa-bajos Butterworth de fase cero
    lowpass_enabled: bool = False
    lowpass_cutoff: float = 150.0  # Hz
    lowpass_order: int = 4


def neighbor_diff_filter(x: np.ndarray,
                         threshold_sigmas: float = 4.0,
                         neighbor_agree_ratio: float = 0.5,
                         max_passes: int = 2
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filtro anti-spike basado en la comparación con vecinos inmediatos.

    Para cada muestra interior x[i], se calculan dos diferencias:
        delta_central   = |x[i] − (x[i−1] + x[i+1]) / 2|     (qué tan lejos está del promedio)
        delta_neighbors = |x[i+1] − x[i−1]|                  (qué tanto difieren los vecinos)

    Una muestra se considera spike si cumple AMBAS condiciones:
        (1) delta_central   > threshold_eff
            └ está lejos de sus vecinos por encima del ruido típico
              (threshold_eff = threshold_sigmas · 1.4826 · MAD(diff(x)))
        (2) delta_neighbors < delta_central · neighbor_agree_ratio
            └ los vecinos están RELATIVAMENTE de acuerdo entre sí
              (i.e. mucho más cerca el uno del otro que del central)

    Esta condición relativa es la clave: invariante al tamaño absoluto del
    spike, tolera la variación natural entre muestras vecinas y solo dispara
    cuando hay una asimetría clara (un salto que "regresa").

    Ejemplo del usuario:  3.00 → 8.00 → 3.12
        delta_central   = |8.00 − (3.00+3.12)/2| = 4.94 bar
        delta_neighbors = |3.12 − 3.00|          = 0.12 bar
        ¿0.12 < 4.94 · 0.5 = 2.47? SÍ → spike, se reemplaza por 3.06.

    En cambio, el flanco de subida del golpe de ariete (p. ej. 2.5 → 3.5 → 4.5)
        delta_central   = |3.5 − 3.5| = 0
        delta_neighbors = |4.5 − 2.5| = 2.0
        ¿2.0 < 0 · 0.5 = 0? NO → preservado, no se toca.

    Múltiples pasadas permiten capturar spikes de 2-3 muestras consecutivas:
    tras la primera pasada los vecinos son samples ya limpios, así que la
    siguiente pasada puede atacar a los que sobrevivieron.

    Returns
    -------
    x_filtered  : np.ndarray
    outlier_mask: np.ndarray (bool) — acumulada de todas las pasadas.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return x.copy(), np.zeros_like(x, dtype=bool)

    # Estimación robusta del nivel de ruido sample-to-sample.
    # El MAD de las diferencias consecutivas no se ve afectado por unos
    # pocos spikes (que aparecen como diffs extremos pero la mediana los ignora).
    diffs = np.diff(x)
    mad = np.median(np.abs(diffs - np.median(diffs)))
    noise_std = max(float(mad) * 1.4826, 1e-9)
    threshold_eff = threshold_sigmas * noise_std

    out = x.copy()
    cumulative_mask = np.zeros_like(x, dtype=bool)

    for _ in range(max(1, int(max_passes))):
        left   = out[:-2]
        right  = out[2:]
        center = out[1:-1]
        avg    = 0.5 * (left + right)

        delta_central   = np.abs(center - avg)
        delta_neighbors = np.abs(right - left)

        # (1) absoluta: el central está lejos de sus vecinos
        cond_far   = delta_central > threshold_eff
        # (2) relativa: los vecinos están "mucho más" cerca entre sí que del central
        cond_agree = delta_neighbors < delta_central * neighbor_agree_ratio

        spike_inner = cond_far & cond_agree

        if not spike_inner.any():
            break

        # Reemplazo por interpolación de vecinos (= promedio de los dos)
        new_center = np.where(spike_inner, avg, center)
        out_new = out.copy()
        out_new[1:-1] = new_center
        out = out_new

        # Mask en coordenadas de la señal completa
        full_mask = np.zeros_like(x, dtype=bool)
        full_mask[1:-1] = spike_inner
        cumulative_mask |= full_mask

    return out, cumulative_mask


def hampel_filter(x: np.ndarray,
                  window_size: int = 7,
                  n_sigmas: float = 3.0
                  ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filtro de Hampel — supresor robusto de picos impulsivos.

    Para cada muestra calcula la mediana y el MAD de una ventana centrada.
    Si la muestra se desvía más de `n_sigmas * 1.4826 * MAD` respecto de la
    mediana local, se reemplaza por esa mediana.

    Conserva el transiente del golpe de ariete (sus muestras forman una
    secuencia coherente que la mediana sigue), mientras que los spikes
    aislados de 1–3 muestras quedan como outliers extremos y se suprimen.

    Returns
    -------
    x_filtered  : np.ndarray
    outlier_mask: np.ndarray (bool) — True donde se sustituyó.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 3:
        return x.copy(), np.zeros_like(x, dtype=bool)
    if window_size % 2 == 0:
        window_size += 1

    s = pd.Series(x)
    rolling_med = s.rolling(window_size, center=True, min_periods=1).median()
    diff = (s - rolling_med).abs()
    rolling_mad = diff.rolling(window_size, center=True, min_periods=1).median()
    threshold = n_sigmas * 1.4826 * rolling_mad

    outlier_mask = ((diff > threshold) & (rolling_mad > 0)).to_numpy()
    x_filt = np.where(outlier_mask, rolling_med.to_numpy(), x)
    return x_filt, outlier_mask


def butterworth_lowpass(x: np.ndarray, fs: float,
                        cutoff_hz: float = 150.0, order: int = 4) -> np.ndarray:
    """Pasa-bajos Butterworth de fase cero (filtfilt)."""
    x = np.asarray(x, dtype=float)
    nyq = 0.5 * fs
    normal_cutoff = cutoff_hz / nyq
    if normal_cutoff >= 0.99 or len(x) < 9:
        return x.copy()
    b, a = sp_signal.butter(order, normal_cutoff, btype="low", analog=False)
    padlen = min(len(x) - 1, 3 * max(len(a), len(b)))
    return sp_signal.filtfilt(b, a, x, padlen=padlen)


def iqr_envelope_filter(x: np.ndarray,
                        window_size: int = 31,
                        n_sigmas: float = 3.0,
                        max_passes: int = 3,
                        upper_pct: float = 75.0,
                        lower_pct: float = 25.0,
                        protect_transients: bool = True,
                        protect_window: int = 101
                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filtro de envolvente por percentiles (rolling-IQR) iterativo.

    Está diseñado para los casos donde Hampel y Vecinos NO bastan: spikes
    que vienen en RACIMOS densos de 4-7 muestras consecutivas anómalas
    (típico de algunos sensores Arduino con ráfagas de ruido). En esos
    casos, una ventana corta para el cálculo de la mediana se contamina,
    pero un percentil 75 sobre una ventana ancha sigue siendo robusto
    porque los spikes son minoría incluso siendo anchos.

    Algoritmo:
        Para cada muestra, en una ventana centrada de tamaño `window_size`:
            Q1 = percentil `lower_pct`
            Q3 = percentil `upper_pct`
            IQR = Q3 - Q1
        Una muestra es outlier si:
            x[i] > Q3 + k·IQR    (spike hacia arriba)
            x[i] < Q1 - k·IQR    (spike hacia abajo)
        El reemplazo se hace por la mediana móvil de la misma ventana
        (que es robusta a outliers).
        Iterar hasta convergencia (no se detectan más outliers).

    Protección de transientes (`protect_transients=True`):
        Calcula la mediana de una ventana muy ancha (`protect_window`,
        default 101 muestras) y su derivada local. Las regiones donde la
        derivada supera el percentil 95 globalmente se marcan como zona de
        transiente real (golpe de ariete, cierre, etc.) y NO se filtran,
        preservando los flancos y los picos legítimos del transiente.

    Parámetros recomendados (a 200-500 Hz):
        window_size=31, n_sigmas=3.0, max_passes=3, protect_window=101

    Returns
    -------
    x_filtered    : np.ndarray
    outlier_mask  : np.ndarray (bool) — acumulada de todas las pasadas.
    protect_mask  : np.ndarray (bool) — máscara de zonas protegidas
                                        (informativa, para visualización).
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 5:
        return x.copy(), np.zeros_like(x, dtype=bool), np.zeros_like(x, dtype=bool)
    if window_size % 2 == 0:
        window_size += 1
    if protect_window % 2 == 0:
        protect_window += 1

    # Máscara de protección de zonas de transiente
    if protect_transients:
        s_orig = pd.Series(x)
        wide_med = s_orig.rolling(protect_window, center=True, min_periods=1).median()
        wide_diff = wide_med.diff().abs()
        # Auto-umbral: percentil 95 de la derivada de la mediana ancha
        # (las zonas de cambio rápido son el 5% más extremo)
        valid = wide_diff.dropna()
        if len(valid) > 0:
            thresh = float(np.quantile(valid, 0.95))
        else:
            thresh = 0.0
        transient = (wide_diff.to_numpy() > thresh) & (thresh > 1e-9)
        # Dilatar la máscara para cubrir el flanco completo
        pad = max(window_size, 21)
        kernel = np.ones(pad, dtype=bool)
        protect_mask = np.convolve(transient.astype(float), kernel.astype(float),
                                    mode='same') > 0
    else:
        protect_mask = np.zeros_like(x, dtype=bool)

    out = x.copy()
    cum = np.zeros_like(x, dtype=bool)

    for _ in range(max(1, int(max_passes))):
        s = pd.Series(out)
        q3 = s.rolling(window_size, center=True, min_periods=1).quantile(upper_pct/100).to_numpy()
        q1 = s.rolling(window_size, center=True, min_periods=1).quantile(lower_pct/100).to_numpy()
        med = s.rolling(window_size, center=True, min_periods=1).median().to_numpy()
        iqr = q3 - q1
        upper = q3 + n_sigmas * iqr
        lower = q1 - n_sigmas * iqr
        mask = ((out > upper) | (out < lower)) & (iqr > 1e-9) & (~protect_mask)
        if not mask.any():
            break
        out = np.where(mask, med, out)
        cum |= mask

    return out, cum, protect_mask


def duration_aware_filter(x: np.ndarray,
                          fs: float,
                          baseline_window: int = 401,
                          k_sigmas: float = 3.5,
                          min_transient_duration_s: float = 0.05,
                          max_passes: int = 3
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filtro consciente de la DURACIÓN: distingue spikes (cortos) de
    transientes reales (largos) por la duración de la anomalía.

    Es la última línea de defensa contra spikes que escapan a los filtros
    anteriores, especialmente cuando vienen agrupados o dentro de una zona
    con cierto contenido frecuencial. La idea es física:

        • Un spike de Arduino dura 1-3 muestras (≈ 3-10 ms).
        • Un golpe de ariete real dura decenas o cientos de ms (>50 ms),
          con muchas muestras consecutivas elevadas.

    Algoritmo:
    1. Calcula una "base" = mediana de ventana muy ANCHA (≈1.2 s).
       Esta base es robusta y representa la "presión esperada" en cada
       momento, ignorando transientes y spikes.
    2. Calcula el residuo |x − base| y un umbral robusto basado en el
       MAD del 85% más bajo del residuo (excluye flancos del transiente).
    3. Identifica grupos de muestras consecutivas con |residuo| > umbral.
    4. Si un grupo dura menos de `min_transient_duration_s` segundos →
       SPIKE → reemplazar por la base.
    5. Si un grupo dura más → TRANSIENTE REAL → preservar tal cual.
    6. Iterar (la base se recalcula entre pasadas).

    Esta es la estrategia más natural: en lugar de razonar sobre
    estadística local (que se contamina con racimos densos), razona
    sobre tiempo. Un usuario podría decir lo mismo mirando el gráfico:
    "este pico dura solo 1 muestra → es ruido; este otro dura 100 ms →
    es real".

    Parámetros recomendados:
        baseline_window: 401 (≈1.2 s a 333 Hz, ≈0.85 s a 471 Hz)
        k_sigmas: 3.5 (3.0 más agresivo, 4.5 más conservador)
        min_transient_duration_s: 0.05 s (50 ms; ningún spike Arduino
                                          típico llega a 30 ms)

    Returns
    -------
    x_filtered : np.ndarray
    spike_mask : np.ndarray (bool) — muestras eliminadas como spike.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 5:
        return x.copy(), np.zeros_like(x, dtype=bool)
    if baseline_window % 2 == 0:
        baseline_window += 1
    min_samples = max(2, int(round(min_transient_duration_s * fs)))

    out = x.copy()
    cum = np.zeros_like(x, dtype=bool)

    for _ in range(max(1, int(max_passes))):
        s = pd.Series(out)
        base = s.rolling(baseline_window, center=True, min_periods=1).median().to_numpy()
        residual = out - base
        abs_res = np.abs(residual)

        # Estimar la dispersión usando solo el 85% más bajo del residuo:
        # esto excluye automáticamente las zonas de flanco real, evitando
        # que la estimación se contamine con la energía del transiente.
        cutoff = float(np.quantile(abs_res, 0.85))
        stable = abs_res[abs_res <= cutoff]
        if len(stable) > 0:
            sigma_robust = max(1.4826 * float(np.median(stable)), 1e-9)
        else:
            sigma_robust = max(float(np.std(abs_res)), 1e-9)
        threshold = k_sigmas * sigma_robust

        # Detectar grupos consecutivos donde |residuo| > umbral
        anomaly = abs_res > threshold
        if not anomaly.any():
            break

        edges = np.diff(np.concatenate([[0], anomaly.astype(int), [0]]))
        starts = np.where(edges == 1)[0]
        ends   = np.where(edges == -1)[0]

        new_mask = np.zeros_like(x, dtype=bool)
        any_short = False
        for s_idx, e_idx in zip(starts, ends):
            if e_idx - s_idx < min_samples:
                # Spike → marcar para limpieza (mantiene transientes intactos)
                new_mask[s_idx:e_idx] = True
                any_short = True

        if not any_short:
            break
        out = np.where(new_mask, base, out)
        cum |= new_mask

    return out, cum


def manual_interval_filter(t: np.ndarray,
                           x: np.ndarray,
                           intervals: List[ManualInterval],
                           local_window: int = 31
                           ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Eliminación manual de picos por intervalo de tiempo.

    Para cada intervalo activo `[t_start, t_end]` con un umbral `thr` y
    un modo `m`:
        - Si m == ">": marca como outlier toda muestra con x[i] > thr
                       cuando t_start <= t[i] <= t_end.
        - Si m == "<": marca como outlier toda muestra con x[i] < thr
                       cuando t_start <= t[i] <= t_end.

    Las muestras marcadas se reemplazan por interpolación lineal entre
    las muestras sanas vecinas (las que están FUERA del conjunto a
    eliminar). Esto es equivalente a "borrar los puntos malos y unir
    los flancos buenos con una recta".

    NOTA SOBRE EL REEMPLAZO (importante):
        En versiones previas se usaba una mediana móvil de ventana fija
        (`local_window`). Esto fallaba silenciosamente cuando había
        muchas muestras outlier consecutivas: si el pico cubría más
        muestras que la ventana, la mediana de cualquier ventana
        centrada en el pico contenía el pico mismo, y el "reemplazo"
        terminaba siendo idéntico al valor original.
        Síntoma típico: la GUI mostraba "N picos suprimidos" con N
        igual al ancho de la ventana, pero la señal no cambiaba en
        absoluto.
        La solución actual usa interpolación lineal entre los puntos
        sanos del entorno; siempre funciona porque los puntos sanos
        están, por construcción, fuera de la zona del pico.

    Parameters
    ----------
    t : np.ndarray
        Vector de tiempo (s).
    x : np.ndarray
        Vector de presión (bar).
    intervals : List[ManualInterval]
        Lista de intervalos a aplicar. Solo los que tienen
        `enabled=True` se procesan.
    local_window : int
        Mantenido por compatibilidad — actualmente no se usa (la
        interpolación lineal no necesita ventana).

    Returns
    -------
    x_filtered : np.ndarray
    mask : np.ndarray (bool) — muestras eliminadas.
    """
    x = np.asarray(x, dtype=float)
    t = np.asarray(t, dtype=float)
    out = x.copy()
    cum = np.zeros_like(x, dtype=bool)

    if not intervals or len(x) < 3:
        return out, cum

    # Acumular máscara TOTAL de todos los intervalos antes de reemplazar.
    # Hacerlo primero asegura que la base de interpolación se calcula
    # sobre la señal limpia, sin contaminación de otros intervalos.
    for itv in intervals:
        if not getattr(itv, "enabled", True):
            continue
        if itv.t_end <= itv.t_start:
            continue
        zone = (t >= itv.t_start) & (t <= itv.t_end)
        if not zone.any():
            continue
        if itv.mode == ">":
            cum |= zone & (x > itv.threshold)
        elif itv.mode == "<":
            cum |= zone & (x < itv.threshold)
        # Cualquier otro modo es ignorado en silencio

    if not cum.any():
        return out, cum

    # Reemplazo por interpolación lineal usando los puntos SANOS como ancla.
    # Marcamos los outliers como NaN y dejamos que pandas interpole.
    x_with_holes = x.copy().astype(float)
    x_with_holes[cum] = np.nan
    filled = pd.Series(x_with_holes).interpolate(
        method="linear", limit_direction="both"
    ).to_numpy()

    # Caso degenerado: si la señal entera era outlier (improbable pero
    # posible si el usuario abusa de los intervalos), pandas devuelve
    # todo NaN. En ese caso usamos la media de la señal original como
    # fallback constante.
    if np.any(np.isnan(filled)):
        fallback = float(np.nanmean(x)) if np.any(~np.isnan(x)) else 0.0
        filled = np.where(np.isnan(filled), fallback, filled)

    out[cum] = filled[cum]
    return out, cum


def apply_filter_pipeline(p: np.ndarray, fs: int, cfg: FilterConfig,
                          t: Optional[np.ndarray] = None
                          ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Pipeline de limpieza en cascada (6 etapas):
        1) Diferencia con vecinos → anti-spike quirúrgico de 1 muestra
                                    (refleja exactamente la lógica
                                    "valor anterior vs valor siguiente").
        2) Hampel                 → mediana móvil + MAD; captura spikes
                                    de 2-3 muestras y residuos.
        3) Envolvente IQR         → suprime racimos densos de 4-7 muestras
                                    con protección automática de los
                                    transientes reales (flancos del
                                    golpe de ariete).
        4) Filtro de duración     → distingue spikes (cortos) de transientes
                                    reales (largos) por la duración. Última
                                    línea de defensa contra picos que
                                    escapan a las etapas anteriores.
        5) Eliminación manual     → el usuario define intervalos [t1, t2]
                                    con un umbral de presión, y se eliminan
                                    los picos que cumplen el criterio en
                                    cada intervalo.
        6) Butterworth LP         → atenúa ruido HF residual sin desfase
                                    (filtfilt).

    Si `t` es None, se construye automáticamente como `np.arange(len(p))/fs`.

    Devuelve la señal final y un dict con etapas intermedias y máscaras
    de outliers de cada etapa por separado, para poder visualizarlas.
    """
    diag: Dict[str, np.ndarray] = {"raw": p.copy()}
    out = p.copy()

    if t is None:
        t = np.arange(len(p), dtype=float) / float(fs)

    if cfg.enabled and cfg.neighbor_enabled:
        out, mask_n = neighbor_diff_filter(
            out, cfg.neighbor_threshold_sigmas,
            cfg.neighbor_agree_ratio, cfg.neighbor_max_passes
        )
        diag["neighbor_outliers"] = mask_n
        diag["after_neighbor"] = out.copy()

    if cfg.enabled and cfg.hampel_enabled:
        out, mask_h = hampel_filter(out, cfg.hampel_window, cfg.hampel_n_sigmas)
        diag["hampel_outliers"] = mask_h
        diag["after_hampel"] = out.copy()

    if cfg.enabled and cfg.iqr_enabled:
        out, mask_iqr, prot_mask = iqr_envelope_filter(
            out,
            window_size=cfg.iqr_window, n_sigmas=cfg.iqr_k,
            max_passes=cfg.iqr_max_passes,
            protect_transients=cfg.iqr_protect_transients,
            protect_window=cfg.iqr_protect_window,
        )
        diag["iqr_outliers"] = mask_iqr
        diag["iqr_protect"] = prot_mask
        diag["after_iqr"] = out.copy()

    if cfg.enabled and cfg.duration_enabled:
        out, mask_dur = duration_aware_filter(
            out, fs,
            baseline_window=cfg.duration_baseline_window,
            k_sigmas=cfg.duration_k_sigmas,
            min_transient_duration_s=cfg.duration_min_transient_s,
            max_passes=cfg.duration_max_passes,
        )
        diag["duration_outliers"] = mask_dur
        diag["after_duration"] = out.copy()

    if cfg.enabled and cfg.manual_enabled and cfg.manual_intervals:
        out, mask_man = manual_interval_filter(t, out, cfg.manual_intervals)
        diag["manual_outliers"] = mask_man
        diag["after_manual"] = out.copy()

    if cfg.enabled and cfg.lowpass_enabled:
        out = butterworth_lowpass(out, fs, cfg.lowpass_cutoff, cfg.lowpass_order)
        diag["after_lowpass"] = out.copy()

    # Máscara combinada de todos los outliers (para visualización rápida)
    combined = np.zeros_like(p, dtype=bool)
    if "neighbor_outliers" in diag:
        combined |= diag["neighbor_outliers"]
    if "hampel_outliers" in diag:
        combined |= diag["hampel_outliers"]
    if "iqr_outliers" in diag:
        combined |= diag["iqr_outliers"]
    if "duration_outliers" in diag:
        combined |= diag["duration_outliers"]
    if "manual_outliers" in diag:
        combined |= diag["manual_outliers"]
    diag["combined_outliers"] = combined

    diag["filtered"] = out
    return out, diag


# ============================================================================
# 4. CARGA DE CSV
# ============================================================================

def load_csv_signal(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carga (t, p) desde un archivo de texto (CSV o log de Arduino/PuTTY).

    El parser es deliberadamente PERMISIVO: lee el archivo línea por
    línea y descarta cualquiera que no contenga exactamente dos números
    parseables. De este modo soporta:

        • CSV "limpio" sin cabecera:
              0.000,0.506
              0.011,0.518

        • CSV con cabecera y/o nombres alternativos:
              time_s,pressure_bar
              0.000,1.016

        • Logs de PuTTY/serial con basura al principio y/o al final:
              =~=~=~=~ PuTTY log 2026... =~=~=~=~
              time_s,pressure_bar
              0.000,1.016
              ...
              20.000,1.065
              FIN

        • Distintos separadores: coma, punto y coma, tabulación o espacios.
        • Líneas vacías y comentarios (# ó // al inicio).

    Solo se conservan líneas que parseen exactamente como
    `<float><sep><float>`. Líneas con más de dos campos (u otra
    estructura) también se descartan, para no contaminar la señal.

    Si el archivo no contiene ninguna línea de datos válida, lanza
    ValueError con un mensaje informativo.
    """
    SEPARATORS = (",", ";", "\t")

    def _parse_pair(line: str):
        """Devuelve (t, p) o None si la línea no es un par de floats."""
        s = line.strip()
        if not s:
            return None
        # Comentarios típicos
        if s[0] in ("#",) or s.startswith("//"):
            return None

        # Probar separadores explícitos primero
        parts = None
        for sep in SEPARATORS:
            if sep in s:
                parts = [x.strip() for x in s.split(sep) if x.strip() != ""]
                break
        if parts is None:
            # Sin separador conocido → probar espacios múltiples
            parts = s.split()

        if len(parts) != 2:
            return None
        try:
            return float(parts[0]), float(parts[1])
        except ValueError:
            return None

    times: List[float] = []
    pressures: List[float] = []
    encodings = ("utf-8", "utf-8-sig", "latin-1")
    last_err: Optional[Exception] = None

    for enc in encodings:
        try:
            with open(path, "r", encoding=enc, errors="strict") as fh:
                for line in fh:
                    pair = _parse_pair(line)
                    if pair is None:
                        continue
                    times.append(pair[0])
                    pressures.append(pair[1])
            break  # leído correctamente con esta codificación
        except UnicodeDecodeError as e:
            last_err = e
            times.clear(); pressures.clear()
            continue
    else:
        # Ningún encoding funcionó
        raise ValueError(
            f"No se pudo decodificar el archivo: {os.path.basename(path)} "
            f"({last_err})"
        )

    if len(times) < 2:
        raise ValueError(
            f"No se encontraron datos numéricos válidos en "
            f"{os.path.basename(path)}. El archivo debe contener al menos\n"
            "dos líneas con el formato:  <tiempo>,<presión>"
        )

    t = np.asarray(times, dtype=float)
    p = np.asarray(pressures, dtype=float)

    # Si los timestamps no son monótonamente crecientes (caso patológico),
    # ordenarlos para evitar problemas posteriores con fs e infer_fs.
    if not np.all(np.diff(t) >= 0):
        order = np.argsort(t, kind="mergesort")
        t, p = t[order], p[order]

    return t, p


def infer_fs(t: np.ndarray) -> int:
    if len(t) < 2:
        return 1000
    dt = float(np.mean(np.diff(t)))
    return int(round(1.0 / dt)) if dt > 0 else 1000


def suggest_filter_params(t: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """
    Analiza una señal y sugiere parámetros para el pipeline de filtrado.

    El análisis es heurístico pero se basa en métricas físicas claras:

    • fs               → frecuencia de muestreo real de la señal.
    • Nivel de ruido   → MAD de las diferencias muestra a muestra
                         (excluye los flancos del transiente).
    • Densidad de spikes → fracción de muestras cuyo |residuo respecto
                           a una mediana ancha| supera 5σ. Si es alta,
                           el ventaneo se reduce y k se vuelve más
                           agresivo. Si es baja, parámetros conservadores.
    • Duración mínima de transiente → calculado a partir de fs para que
                                      ≈30 ms sea siempre ≥ 5 muestras.
    • Cutoff Butterworth → 4·f_dominante en el contenido espectral.

    Devuelve un diccionario con los valores sugeridos para CADA control
    expuesto en la UI. Si la señal es demasiado corta o degenerada,
    devuelve los defaults seguros del FilterConfig.
    """
    suggestions: Dict[str, float] = {}
    p = np.asarray(p, dtype=float)
    t = np.asarray(t, dtype=float)
    n = len(p)
    if n < 50:
        return suggestions

    # ── Frecuencia de muestreo ──────────────────────────────────
    fs = float(infer_fs(t))
    suggestions["fs"] = fs

    # ── Estimar ruido típico (MAD de Δp robusto a outliers) ────
    diffs = np.diff(p)
    mad_diff = float(np.median(np.abs(diffs - np.median(diffs))))
    sigma_noise = max(mad_diff * 1.4826, 1e-9)
    suggestions["sigma_noise"] = sigma_noise

    # ── Mediana ancha como base + densidad de spikes ───────────
    # Ventana ≈ 1 s con tope mínimo
    base_w = max(31, min(401, int(round(fs * 1.0)) | 1))
    if base_w % 2 == 0:
        base_w += 1
    s = pd.Series(p)
    base = s.rolling(base_w, center=True, min_periods=1).median().to_numpy()
    abs_res = np.abs(p - base)
    cutoff = float(np.quantile(abs_res, 0.85))
    stable = abs_res[abs_res <= cutoff]
    if len(stable) > 0:
        sigma_robust = max(1.4826 * float(np.median(stable)), 1e-9)
    else:
        sigma_robust = sigma_noise
    spike_mask = abs_res > 5.0 * sigma_robust
    spike_density = float(np.sum(spike_mask)) / float(n)  # fracción 0..1
    suggestions["spike_density"] = spike_density

    # ── 1) Vecinos ──────────────────────────────────────────────
    # Si hay muchos spikes, mantener n·σ algo bajo (4.0); si la señal
    # es limpia, subir a 5.0 para ser conservador. Siempre ≥3.5.
    if spike_density > 0.05:
        suggestions["neighbor_n_sigmas"] = 3.5
        suggestions["neighbor_passes"] = 3
    elif spike_density > 0.01:
        suggestions["neighbor_n_sigmas"] = 4.0
        suggestions["neighbor_passes"] = 2
    else:
        suggestions["neighbor_n_sigmas"] = 5.0
        suggestions["neighbor_passes"] = 2

    # ── 2) Hampel ───────────────────────────────────────────────
    # Ventana adaptada a fs: cubrir ≈15 ms (impar). Mínimo 5, máx 21.
    h_win = max(5, min(21, int(round(fs * 0.015)) | 1))
    if h_win % 2 == 0:
        h_win += 1
    suggestions["hampel_window"] = h_win
    suggestions["hampel_n_sigmas"] = 3.0 if spike_density > 0.005 else 3.5

    # ── 3) IQR ──────────────────────────────────────────────────
    # Ventana ≈ 70 ms (impar). Suficiente para que racimos de 4-7
    # muestras sigan siendo minoría.
    iqr_win = max(11, min(61, int(round(fs * 0.07)) | 1))
    if iqr_win % 2 == 0:
        iqr_win += 1
    suggestions["iqr_window"] = iqr_win
    suggestions["iqr_k"] = 2.5 if spike_density > 0.05 else 3.0
    suggestions["iqr_passes"] = 5 if spike_density > 0.05 else 3

    # ── 4) Duración ─────────────────────────────────────────────
    # Ventana ancha ≈ 1.2 s; mín_dur = 50 ms (típico Arduino)
    dur_w = max(101, min(801, int(round(fs * 1.2)) | 1))
    if dur_w % 2 == 0:
        dur_w += 1
    suggestions["dur_baseline"] = dur_w
    suggestions["dur_k"] = 3.0 if spike_density > 0.05 else 3.5
    # Para fs muy alto, 50 ms son muchas muestras (puede sobrevivir
    # un transiente real corto). Ajuste suave:
    suggestions["dur_min"] = 0.05 if fs >= 200 else 0.08
    suggestions["dur_passes"] = 5 if spike_density > 0.05 else 3

    # ── 6) Butterworth ──────────────────────────────────────────
    # Frecuencia dominante del transiente: pico de la PSD
    # excluyendo DC y por debajo de fs/4 para evitar capturar HF.
    try:
        baseline_dc = np.median(p[: max(1, int(0.05 * n))])
        f_psd, Pxx = sp_signal.welch(p - baseline_dc, fs=fs,
                                      nperseg=min(1024, n))
        # Limitar al rango útil
        f_max_useful = fs * 0.4
        sel = (f_psd > 1.0) & (f_psd < f_max_useful)
        if sel.any():
            f_dom = float(f_psd[sel][np.argmax(Pxx[sel])])
            cutoff_lp = float(np.clip(4.0 * f_dom, 30.0, fs * 0.4))
        else:
            cutoff_lp = float(np.clip(fs / 6.0, 30.0, 500.0))
    except Exception:
        cutoff_lp = float(np.clip(fs / 6.0, 30.0, 500.0))
    suggestions["lowpass_cutoff"] = round(cutoff_lp, 1)

    return suggestions


def suggest_training_params(
    n_normal: int,
    n_bypass: int,
    fs_list: List[int],
    duration_list: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Sugiere parámetros de data-augmentation y de ML adaptados al dataset
    real cargado, con el objetivo de maximizar la robustez del modelo.

    El razonamiento heurístico es el siguiente:

    DATASET PEQUEÑO (≤30 señales por clase):
        El riesgo principal es el sobreajuste. La estrategia es:
        • Más aumentaciones por señal (10-15) para diversificar.
        • Tamaño final del dataset relativamente grande (≈n_min · 2 · 12).
        • RF con menos árboles pero profundos (overfit menos rápido en
          datasets pequeños) y test_size más grande (0.3) para tener
          una validación fiable.
        • SVM con C bajo (0.5) para regularización fuerte.

    DATASET MEDIO (30-100 señales por clase):
        • 5-8 aumentaciones.
        • Tamaño final 600-1200 muestras.
        • RF con 200-300 árboles, test_size 0.25, SVM C=1.0.

    DATASET GRANDE (>100 señales por clase):
        • 2-4 aumentaciones (suficiente, evita explotar el dataset).
        • Tamaño final ≈ 2·n_total (no más).
        • RF con más árboles (300-500), test_size 0.2, SVM C=2.0.

    Frecuencia de muestreo:
        Si la `fs` mediana es alta (>1000 Hz), las features wavelet/PSD
        son más ricas → SVM tiende a aprovecharlo mejor con C ligeramente
        mayor; RF se beneficia de más árboles.

    Desbalance de clases:
        Si una clase tiene ≥3× la otra, se fuerza un `target_total` que
        rebalancee a 50/50, y se sube algo el n_aug para la clase
        minoritaria (esto el GUI lo verá como aumentaciones ‘más útiles’).

    Returns
    -------
    Dict con claves:
        n_aug, target_total, n_estimators, svm_c, test_size, use_rf, use_svm
        + 'reasoning': str explicativo para mostrar al usuario.
    """
    n_min = min(n_normal, n_bypass) if (n_normal and n_bypass) else max(n_normal, n_bypass)
    n_total_real = max(1, n_normal + n_bypass)
    fs_med = float(np.median(fs_list)) if fs_list else 1000.0
    dur_med = float(np.median(duration_list)) if duration_list else 5.0

    out: Dict[str, Any] = {}
    reason: List[str] = []
    reason.append(f"Dataset real: {n_normal} Normal · {n_bypass} Bypass "
                  f"(min/clase = {n_min}).")
    reason.append(f"fs mediana = {fs_med:.0f} Hz, duración mediana = {dur_med:.2f} s.")

    # — Aumentaciones por señal —
    if n_min <= 10:
        n_aug = 15
        reason.append(f"  ◦ Dataset MUY pequeño → n_aug=15 (máxima diversificación).")
    elif n_min <= 30:
        n_aug = 10
        reason.append(f"  ◦ Dataset pequeño → n_aug=10.")
    elif n_min <= 100:
        n_aug = 6
        reason.append(f"  ◦ Dataset medio → n_aug=6 (default robusto).")
    elif n_min <= 300:
        n_aug = 4
        reason.append(f"  ◦ Dataset grande → n_aug=4.")
    else:
        n_aug = 2
        reason.append(f"  ◦ Dataset muy grande → n_aug=2 (no inflar más).")
    out["n_aug"] = n_aug

    # — Tamaño final del dataset —
    # Multiplicador adaptado al tamaño real
    desired = n_total_real * (n_aug + 1)  # original + n_aug copias
    # Lo redondeamos hacia múltiplo de 2 para que sea balanceable
    desired = max(40, ((desired + 1) // 2) * 2)
    # Topes razonables
    desired = int(np.clip(desired, 40, 8000))
    out["target_total"] = desired
    reason.append(f"  ◦ Dataset final ≈ {desired} (n_total · (n_aug+1) clip 40..8000).")

    # — Random Forest: nº de árboles —
    if n_min <= 30:
        n_estimators = 100
    elif n_min <= 100:
        n_estimators = 200
    elif n_min <= 300:
        n_estimators = 300
    else:
        n_estimators = 400
    # Boost adicional si la fs es alta (más features útiles)
    if fs_med >= 1000:
        n_estimators = min(500, n_estimators + 50)
        reason.append(f"  ◦ fs alta → +50 árboles RF.")
    out["n_estimators"] = n_estimators

    # — SVM C —
    if n_min <= 20:
        svm_c = 0.5
    elif n_min <= 100:
        svm_c = 1.0
    elif n_min <= 300:
        svm_c = 1.5
    else:
        svm_c = 2.0
    if fs_med >= 1000:
        svm_c = min(5.0, svm_c * 1.25)
    out["svm_c"] = round(float(svm_c), 2)

    # — Test size: con poco dataset queremos test grande para tener
    #   estimación fiable; con mucho, test pequeño para entrenar más.
    if n_min <= 20:
        test_size = 0.30
    elif n_min <= 100:
        test_size = 0.25
    else:
        test_size = 0.20
    out["test_size"] = round(float(test_size), 2)

    # — Modelos a entrenar —
    # Si el dataset es pequeñísimo, deshabilitar SVM (RBF con poco
    # dataset es muy variable). Pero si el usuario lo desea, lo dejamos
    # como sugerencia activable.
    out["use_rf"] = True
    out["use_svm"] = (n_min >= 10)
    if not out["use_svm"]:
        reason.append("  ◦ SVM desactivado: dataset insuficiente (<10 por clase).")

    # — Aviso de desbalance —
    if n_normal > 0 and n_bypass > 0:
        ratio = max(n_normal, n_bypass) / min(n_normal, n_bypass)
        if ratio >= 3.0:
            reason.append(
                f"  ⚠️  Desbalance fuerte ({ratio:.1f}×). El target_total "
                f"forzará 50/50 vía oversampling de la clase minoritaria."
            )

    out["n_estimators"] = int(out["n_estimators"])
    out["reasoning"] = "\n".join(reason)
    return out


# ============================================================================
# 5. DATASET SINTÉTICO
# ============================================================================

# Rangos de parámetros físicos por defecto al generar datasets sintéticos.
# Cada entrada es (min, max) para una distribución uniforme; `bypass_prob`
# es la probabilidad de generar una señal de clase 1.
DEFAULT_PARAM_RANGES: Dict[str, Tuple[float, float]] = {
    "p0":        (1.8, 3.5),
    "A":         (0.3, 1.0),
    "f0":        (10.0, 70.0),
    "tau":       (0.15, 0.7),
    "noise_std": (0.005, 0.02),
    "t0":        (0.2, 0.8),
}


def generate_dataset(n_samples: int = 400, fs: int = 2000, duration: float = 5.0,
                     seed: int = 42, progress_cb=None,
                     param_ranges: Optional[Dict[str, Tuple[float, float]]] = None,
                     bypass_prob: float = 0.5,
                     ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Genera un dataset sintético de transientes etiquetados.

    Parameters
    ----------
    n_samples : nº total de muestras a generar.
    fs        : frecuencia de muestreo en Hz.
    duration  : duración de cada señal (s).
    seed      : semilla del generador.
    progress_cb : callable opcional para reportar progreso (0-100).
    param_ranges : dict opcional con rangos (min, max) para muestrear los
        parámetros físicos. Claves esperadas: p0, A, f0, tau, noise_std, t0.
        Si una clave falta, se usa el valor de DEFAULT_PARAM_RANGES.
    bypass_prob : probabilidad de generar una señal de clase bypass (1).
    """
    pr = dict(DEFAULT_PARAM_RANGES)
    if param_ranges:
        # Validar y mezclar con defaults
        for k, v in param_ranges.items():
            if k in pr and isinstance(v, (tuple, list)) and len(v) == 2:
                lo, hi = float(v[0]), float(v[1])
                if hi < lo:  # tolerar inversión
                    lo, hi = hi, lo
                pr[k] = (lo, hi)

    X, y = [], []
    rng = np.random.default_rng(seed)
    feature_names: List[str] = []
    for i in range(n_samples):
        params = TransientParams(
            duration=duration, fs=fs,
            p0=rng.uniform(*pr["p0"]),
            A=rng.uniform(*pr["A"]),
            f0=rng.uniform(*pr["f0"]),
            tau=rng.uniform(*pr["tau"]),
            noise_std=rng.uniform(*pr["noise_std"]),
            t0=rng.uniform(*pr["t0"]),
            bypass=bool(rng.random() < bypass_prob),
        )
        t, p = generate_transient(params)
        feats = extract_features(t, p, fs)
        if not feature_names:
            feature_names = sorted(feats.keys())
        X.append([feats[k] for k in feature_names])
        y.append(int(params.bypass))
        if progress_cb is not None and (i % max(1, n_samples // 50) == 0):
            progress_cb(int(100 * (i + 1) / n_samples))
    if progress_cb is not None:
        progress_cb(100)
    return np.array(X), np.array(y), feature_names


# ============================================================================
# 6. RESULTADOS DE ENTRENAMIENTO Y WORKERS
# ============================================================================

@dataclass
class TrainingResult:
    """
    Resultado completo de un entrenamiento o re-evaluación.

    En v3.14 se ha generalizado para soportar N modelos. Los datos
    primarios viven en `models`, `metrics`, `cms`, `cvs` y `reports`,
    todos indexados por la clave del modelo (rf/svm/xgb/lgbm).

    Los campos legacy `rf`, `svm`, `accuracy_rf`, ... se mantienen como
    properties que delegan en los nuevos diccionarios, para que el
    código que carga .joblib viejos siga funcionando sin cambios.
    """
    # Modelos entrenados, indexados por clave (rf, svm, xgb, lgbm).
    # Los valores son los modelos sklearn-compatibles (posiblemente
    # envueltos en CalibratedClassifierCV si la calibración está activa).
    models: Dict[str, Any] = field(default_factory=dict)

    # Scaler único, compartido por todos los modelos.
    scaler: Optional[StandardScaler] = None
    feature_names: List[str] = field(default_factory=list)

    # Métricas por modelo (clave → valor).
    metrics: Dict[str, Optional[float]] = field(default_factory=dict)
    cvs:     Dict[str, Optional[np.ndarray]] = field(default_factory=dict)
    cms:     Dict[str, Optional[np.ndarray]] = field(default_factory=dict)
    reports: Dict[str, str] = field(default_factory=dict)

    # Importancia de features (la entregamos solo si hay un modelo
    # basado en árboles disponible; preferimos RF si existe).
    feature_importance: Optional[np.ndarray] = None

    source: str = "synthetic"   # 'synthetic' | 'real' | 'loaded' | 'reeval' …
    n_samples: int = 0

    # Detalle por muestra. Cada elemento es un dict con:
    #   { 'name': str, 'true': int (0|1),
    #     'pred_rf': int|None,    'prob_rf_bypass': float|None,
    #     'pred_svm': int|None,   'prob_svm_bypass': float|None,
    #     'pred_xgb': int|None,   'prob_xgb_bypass': float|None,
    #     'pred_lgbm': int|None,  'prob_lgbm_bypass': float|None,
    #     'pred_ensemble': int|None,
    #     'prob_ensemble_bypass': float|None }
    per_sample: List[Dict[str, Any]] = field(default_factory=list)

    # ── Properties legacy para retrocompatibilidad ─────────────
    @property
    def rf(self) -> Optional[Any]:
        return self.models.get("rf")
    @rf.setter
    def rf(self, val):
        if val is None:
            self.models.pop("rf", None)
        else:
            self.models["rf"] = val

    @property
    def svm(self) -> Optional[Any]:
        return self.models.get("svm")
    @svm.setter
    def svm(self, val):
        if val is None:
            self.models.pop("svm", None)
        else:
            self.models["svm"] = val

    @property
    def accuracy_rf(self) -> Optional[float]:
        return self.metrics.get("rf")
    @accuracy_rf.setter
    def accuracy_rf(self, val): self.metrics["rf"] = val

    @property
    def accuracy_svm(self) -> Optional[float]:
        return self.metrics.get("svm")
    @accuracy_svm.setter
    def accuracy_svm(self, val): self.metrics["svm"] = val

    @property
    def cv_rf(self) -> Optional[np.ndarray]:
        return self.cvs.get("rf")
    @cv_rf.setter
    def cv_rf(self, val): self.cvs["rf"] = val

    @property
    def cv_svm(self) -> Optional[np.ndarray]:
        return self.cvs.get("svm")
    @cv_svm.setter
    def cv_svm(self, val): self.cvs["svm"] = val

    @property
    def cm_rf(self) -> Optional[np.ndarray]:
        return self.cms.get("rf")
    @cm_rf.setter
    def cm_rf(self, val): self.cms["rf"] = val

    @property
    def cm_svm(self) -> Optional[np.ndarray]:
        return self.cms.get("svm")
    @cm_svm.setter
    def cm_svm(self, val): self.cms["svm"] = val

    @property
    def report_rf(self) -> str:
        return self.reports.get("rf", "")
    @report_rf.setter
    def report_rf(self, val): self.reports["rf"] = val or ""

    @property
    def report_svm(self) -> str:
        return self.reports.get("svm", "")
    @report_svm.setter
    def report_svm(self, val): self.reports["svm"] = val or ""

    # ── Helpers ────────────────────────────────────────────────
    def available_models(self) -> List[str]:
        """Lista de claves de modelos entrenados (en orden canónico)."""
        return [k for k in MODEL_KEYS if self.models.get(k) is not None]

    def has_ensemble(self) -> bool:
        """¿Tiene sentido un voting ensemble? Sí si hay 2+ modelos."""
        return len(self.available_models()) >= 2

    def to_dict(self) -> dict:
        """Serializa las métricas (sin los modelos) para persistir en joblib."""
        return {
            "metrics":           dict(self.metrics),
            "cvs":               dict(self.cvs),
            "cms":               dict(self.cms),
            "reports":           dict(self.reports),
            "feature_importance": self.feature_importance,
            "source":            self.source,
            "n_samples":         self.n_samples,
            "per_sample":        self.per_sample,
            # Campos legacy duplicados para que .joblib v3 puedan abrirse
            # con código v3.13 si alguien hace downgrade.
            "accuracy_rf":  self.metrics.get("rf"),
            "accuracy_svm": self.metrics.get("svm"),
            "cv_rf":  self.cvs.get("rf"),
            "cv_svm": self.cvs.get("svm"),
            "cm_rf":  self.cms.get("rf"),
            "cm_svm": self.cms.get("svm"),
            "report_rf":  self.reports.get("rf", ""),
            "report_svm": self.reports.get("svm", ""),
        }


def _prob_bypass(model, X: np.ndarray) -> Optional[np.ndarray]:
    """
    Devuelve la probabilidad estimada de la clase 1 (bypass) para cada
    fila de X.

    Usa `predict_proba` si el modelo lo expone (RandomForest siempre lo
    tiene; SVC sólo si fue entrenado con `probability=True`). Si no, usa
    `decision_function` y la mapea a [0, 1] con un sigmoide. Si tampoco
    eso está disponible, devuelve None.
    """
    if model is None or len(X) == 0:
        return None
    # Camino preferido: predict_proba
    pp = getattr(model, "predict_proba", None)
    if callable(pp):
        try:
            prob = pp(X)
            classes = list(getattr(model, "classes_", [0, 1]))
            # Localizar el índice de la clase 1
            if 1 in classes:
                col = classes.index(1)
            else:
                col = -1
            return np.asarray(prob[:, col], dtype=float)
        except Exception:
            pass
    # Fallback: decision_function + sigmoide
    df = getattr(model, "decision_function", None)
    if callable(df):
        try:
            d = np.asarray(df(X), dtype=float).ravel()
            return 1.0 / (1.0 + np.exp(-d))
        except Exception:
            pass
    return None


def ensemble_prob_bypass(models: Dict[str, Any],
                          X: np.ndarray) -> Optional[np.ndarray]:
    """
    Soft-voting ensemble: promedia las probabilidades de la clase
    bypass de TODOS los modelos disponibles. Devuelve None si no hay
    al menos dos modelos válidos.
    """
    probs = []
    for key in MODEL_KEYS:
        m = models.get(key)
        if m is None:
            continue
        p = _prob_bypass(m, X)
        if p is not None:
            probs.append(p)
    if len(probs) < 2:
        return None
    return np.mean(np.stack(probs, axis=0), axis=0)


def _make_classifier(key: str,
                     n_estimators: int = 150,
                     svm_c: float = 1.0,
                     learning_rate: float = 0.1,
                     max_depth: int = 6,
                     ):
    """
    Factory: instancia un clasificador sklearn-compatible para la clave
    dada. Lanza ValueError si la dependencia no está instalada.

    Parámetros:
        n_estimators : nº de árboles (RF, XGB, LGBM)
        svm_c        : C de regularización para SVM
        learning_rate: tasa de aprendizaje (XGB, LGBM)
        max_depth    : profundidad máxima de los árboles (XGB, LGBM)
    """
    if key == "rf":
        return RandomForestClassifier(
            # n_jobs=-1 es seguro incluso en .exe congelado: los bosques
            # de sklearn paralelizan con HILOS (prefer="threads"), no
            # con procesos loky.
            n_estimators=n_estimators, random_state=42, n_jobs=-1
        )
    if key == "svm":
        return SVC(kernel="rbf", probability=True, C=svm_c, random_state=42)
    if key == "xgb":
        if not XGBOOST_AVAILABLE:
            raise ValueError(
                "XGBoost no está instalado. Instala con: pip install xgboost"
            )
        # use_label_encoder=False quita un warning antiguo; eval_metric
        # silencia otro y verbosity=0 silencia el log durante fit.
        return XGBClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=42, n_jobs=-1,
            eval_metric="logloss", verbosity=0,
            use_label_encoder=False,
        )
    if key == "lgbm":
        if not LIGHTGBM_AVAILABLE:
            raise ValueError(
                "LightGBM no está instalado. Instala con: pip install lightgbm"
            )
        return LGBMClassifier(
            n_estimators=n_estimators, learning_rate=learning_rate,
            max_depth=max_depth, random_state=42, n_jobs=-1, verbosity=-1,
        )
    raise ValueError(f"Clave de modelo desconocida: {key!r}")


def _calibrated_wrap(model, X_tr: np.ndarray, y_tr: np.ndarray,
                     enable: bool, cv: int = 3):
    """
    Envuelve `model` en CalibratedClassifierCV(method='isotonic') si
    `enable=True` y el dataset es lo bastante grande para los `cv`
    folds que requiere la calibración (cada fold necesita al menos 2
    muestras por clase).

    Devuelve el modelo (calibrado o no) sin entrenar — el caller debe
    hacer fit. Si la calibración no es viable, retorna el modelo sin
    envolver y emite un warning silencioso.
    """
    if not enable:
        return model
    # Verificar que cada clase tenga >= cv muestras (mínimo absoluto).
    _, counts = np.unique(y_tr, return_counts=True)
    if counts.min() < cv:
        # Demasiado pocos ejemplos para calibrar fiablemente
        return model
    return CalibratedClassifierCV(
        # (v4.5) n_jobs con PROCESOS (loky): en .exe congelado debe ser
        # secuencial — ver PROC_N_JOBS arriba.
        estimator=model, method="isotonic", cv=cv, n_jobs=PROC_N_JOBS
    )


def _train_models(X: np.ndarray, y: np.ndarray,
                  *,
                  models_to_train: List[str],
                  n_estimators: int = 150,
                  svm_c: float = 1.0,
                  learning_rate: float = 0.1,
                  max_depth: int = 6,
                  test_size: float = 0.25,
                  calibrate: bool = False,
                  progress_cb=None) -> TrainingResult:
    """
    Entrena varios modelos en el mismo split y devuelve un TrainingResult
    con todos los modelos, métricas, matrices de confusión y CV.

    Si `calibrate=True`, cada modelo se envuelve en CalibratedClassifierCV
    antes del fit. Esto hace que `predict_proba` devuelva probabilidades
    calibradas (más fiables como medidas de confianza), a costa de un
    pequeño coste extra de entrenamiento.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    X_tr, X_te, y_tr, y_te = train_test_split(
        Xs, y, test_size=test_size, random_state=42, stratify=y
    )

    tr = TrainingResult(
        scaler=scaler, feature_names=[], n_samples=len(y),
    )

    # Filtrar lo solicitado a lo realmente disponible
    keys = [k for k in MODEL_KEYS
            if k in models_to_train and model_is_available(k)]
    n_steps = max(1, len(keys))
    feat_imp_source: Optional[Any] = None

    for i, key in enumerate(keys):
        pct = 60 + int(35 * i / n_steps)
        suffix = "+ calibración" if calibrate else ""
        if progress_cb:
            progress_cb(pct, f"Entrenando {MODEL_DISPLAY_NAMES[key]} {suffix}…")

        try:
            base = _make_classifier(
                key, n_estimators=n_estimators, svm_c=svm_c,
                learning_rate=learning_rate, max_depth=max_depth,
            )
        except ValueError as e:
            # Dependencia ausente — saltar
            if progress_cb:
                progress_cb(pct, f"⚠ Saltando {key}: {e}")
            continue

        clf = _calibrated_wrap(base, X_tr, y_tr, enable=calibrate, cv=3)
        clf.fit(X_tr, y_tr)
        tr.models[key] = clf

        # Evaluación en el hold-out
        yp = clf.predict(X_te)
        tr.metrics[key] = accuracy_score(y_te, yp)
        tr.cms[key] = confusion_matrix(y_te, yp, labels=[0, 1])
        tr.reports[key] = classification_report(
            y_te, yp, labels=[0, 1],
            target_names=["Normal", "Bypass"], zero_division=0
        )
        # CV en el dataset completo (estable)
        try:
            # (v4.5) cross_val_score paraleliza con PROCESOS (loky):
            # secuencial en .exe congelado — ver PROC_N_JOBS.
            tr.cvs[key] = cross_val_score(clf, Xs, y, cv=5,
                                          n_jobs=PROC_N_JOBS)
        except Exception:
            tr.cvs[key] = None

        # Importancia: preferir RF (más interpretable), si no XGB, si no LGBM
        if feat_imp_source is None:
            # Si está calibrado, hay que descender al estimador base
            inner = clf
            if isinstance(clf, CalibratedClassifierCV):
                # El primer fold tiene el modelo entrenado
                try:
                    inner = clf.calibrated_classifiers_[0].estimator
                except Exception:
                    inner = None
            if inner is not None and hasattr(inner, "feature_importances_"):
                feat_imp_source = inner.feature_importances_

    tr.feature_importance = feat_imp_source

    if progress_cb:
        progress_cb(100, "Completado.")
    return tr


# Alias retrocompatible — algunos workers viejos podrían llamarlo.
def _train_both(X: np.ndarray, y: np.ndarray,
                n_estimators: int, svm_c: float, test_size: float,
                train_rf: bool = True, train_svm: bool = True,
                progress_cb=None) -> TrainingResult:
    """Wrapper retrocompatible. Prefiere `_train_models` para código nuevo."""
    keys: List[str] = []
    if train_rf:  keys.append("rf")
    if train_svm: keys.append("svm")
    return _train_models(
        X, y, models_to_train=keys,
        n_estimators=n_estimators, svm_c=svm_c, test_size=test_size,
        calibrate=False, progress_cb=progress_cb,
    )


class SyntheticTrainingWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, str)
    finished = QtCore.pyqtSignal(object)
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, n_samples: int,
                 n_estimators: int, svm_c: float,
                 fs: int, duration: float, test_size: float,
                 models_to_train: List[str],
                 calibrate: bool = False,
                 learning_rate: float = 0.1, max_depth: int = 6,
                 param_ranges: Optional[Dict[str, Tuple[float, float]]] = None):
        super().__init__()
        self.n_samples = n_samples
        self.n_estimators = n_estimators
        self.svm_c = svm_c
        self.fs, self.duration, self.test_size = fs, duration, test_size
        self.models_to_train = list(models_to_train)
        self.calibrate = calibrate
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.param_ranges = param_ranges

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.progress.emit(5, "Generando dataset sintético…")
            X, y, names = generate_dataset(
                n_samples=self.n_samples, fs=self.fs, duration=self.duration,
                param_ranges=self.param_ranges,
                progress_cb=lambda p: self.progress.emit(
                    5 + int(p * 0.55), f"Generando dataset… {p}%"
                ),
            )
            result = _train_models(
                X, y,
                models_to_train=self.models_to_train,
                n_estimators=self.n_estimators, svm_c=self.svm_c,
                learning_rate=self.learning_rate, max_depth=self.max_depth,
                test_size=self.test_size,
                calibrate=self.calibrate,
                progress_cb=self.progress.emit,
            )
            result.feature_names = names
            result.source = "synthetic"
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())


class RealTrainingWorker(QtCore.QObject):
    progress = QtCore.pyqtSignal(int, str)
    finished = QtCore.pyqtSignal(object)
    failed   = QtCore.pyqtSignal(str)

    def __init__(self, data_no: list, data_yes: list,
                 n_aug: int, target_total: int,
                 n_estimators: int, svm_c: float, test_size: float,
                 models_to_train: List[str],
                 calibrate: bool = False,
                 learning_rate: float = 0.1, max_depth: int = 6):
        super().__init__()
        self.data_no, self.data_yes = data_no, data_yes
        self.n_aug, self.target_total = n_aug, target_total
        self.n_estimators, self.svm_c, self.test_size = n_estimators, svm_c, test_size
        self.models_to_train = list(models_to_train)
        self.calibrate = calibrate
        self.learning_rate = learning_rate
        self.max_depth = max_depth

    @QtCore.pyqtSlot()
    def run(self):
        try:
            self.progress.emit(5, "Extrayendo features (clase Normal)…")
            feats_no  = self._process_group(self.data_no,  offset=5,  span=25)
            self.progress.emit(35, "Extrayendo features (clase Bypass)…")
            feats_yes = self._process_group(self.data_yes, offset=35, span=25)

            if not feats_no or not feats_yes:
                raise ValueError("No se pudieron extraer características de una de las clases.")

            feature_names = sorted(feats_no[0].keys())
            X_no  = np.array([[f[k] for k in feature_names] for f in feats_no])
            X_yes = np.array([[f[k] for k in feature_names] for f in feats_yes])

            half = self.target_total // 2

            def sample(Xg, N):
                idx = np.random.choice(Xg.shape[0], size=N, replace=(Xg.shape[0] < N))
                return Xg[idx]

            X = np.vstack([sample(X_no, half), sample(X_yes, self.target_total - half)])
            y = np.hstack([np.zeros(half, dtype=int),
                           np.ones(self.target_total - half, dtype=int)])
            perm = np.random.permutation(len(y))
            X, y = X[perm], y[perm]

            self.progress.emit(60, "Entrenando modelos…")
            result = _train_models(
                X, y,
                models_to_train=self.models_to_train,
                n_estimators=self.n_estimators, svm_c=self.svm_c,
                learning_rate=self.learning_rate, max_depth=self.max_depth,
                test_size=self.test_size,
                calibrate=self.calibrate,
                progress_cb=self.progress.emit,
            )
            result.feature_names = feature_names
            result.source = "real"
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _process_group(self, group, offset: int, span: int):
        feats = []
        total = max(1, len(group))
        for i, (path, t, p, fs) in enumerate(group):
            for t2, p2 in augment_single_signal(t, p, n_aug=self.n_aug):
                try:
                    feats.append(extract_features(t2, p2, fs))
                except Exception:
                    pass
            self.progress.emit(offset + int(span * (i + 1) / total),
                               f"Procesando… {os.path.basename(path)}")
        return feats


# ============================================================================
# 6.5. VALIDACIÓN CRUZADA
# ============================================================================

@dataclass
class CrossValidationConfig:
    """Configuración de un run de validación cruzada."""
    # Tipo de split: "kfold" (no estratificado) o "stratified" (recomendado
    # para clasificación, sobre todo si el dataset está desbalanceado).
    strategy: str = "stratified"   # "kfold" | "stratified"
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42
    # Modelos a validar (subconjunto de MODEL_KEYS)
    models_to_validate: List[str] = field(default_factory=list)
    # Hiperparámetros — mismos que el entrenamiento normal
    n_estimators: int = 150
    svm_c: float = 1.0
    learning_rate: float = 0.1
    max_depth: int = 6
    calibrate: bool = True
    # Augmentación de datos previa a la CV
    n_aug: int = 6
    target_total: int = 1200


@dataclass
class FoldMetrics:
    """Métricas de un único fold para un único modelo."""
    fold_index: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: Optional[float]
    confusion: np.ndarray   # shape (2,2)
    train_time_s: float
    n_train: int
    n_val: int


@dataclass
class ModelCVResult:
    """Resultados agregados de CV para un modelo."""
    model_key: str
    folds: List[FoldMetrics] = field(default_factory=list)
    total_time_s: float = 0.0

    # Predicciones por muestra ORIGINAL acumuladas a lo largo de los folds.
    # Cada muestra original i fue "validation" en exactamente un fold; aquí
    # guardamos: pred_per_fold[i] = predicción en su fold de validación
    #            prob_per_fold[i] = prob_bypass en su fold de validación
    # En CV estándar cada muestra aparece UNA vez en validation, así que
    # estos arrays son del tamaño del dataset.
    val_preds: Optional[np.ndarray] = None     # int (0|1)
    val_probs: Optional[np.ndarray] = None     # float [0,1]
    val_truth: Optional[np.ndarray] = None     # int (0|1) — mismas etiquetas para todos los modelos

    # Modelos entrenados en cada fold. fold_models[i] es el modelo sklearn
    # del fold (1-indexed: igual que f.fold_index). fold_scalers[i] es el
    # scaler correspondiente (cada fold tiene su propio scaler para evitar
    # leakage). Estos se usan en la sección de Ranking para permitir al
    # usuario seleccionar un modelo concreto y descargarlo.
    fold_models:  Dict[int, Any] = field(default_factory=dict)
    fold_scalers: Dict[int, Any] = field(default_factory=dict)

    def metric_array(self, attr: str) -> np.ndarray:
        return np.array([getattr(f, attr) for f in self.folds], dtype=float)

    @property
    def mean_accuracy(self) -> float:
        return float(np.mean(self.metric_array("accuracy")))

    @property
    def std_accuracy(self) -> float:
        return float(np.std(self.metric_array("accuracy")))

    @property
    def mean_precision(self) -> float:
        return float(np.mean(self.metric_array("precision")))

    @property
    def mean_recall(self) -> float:
        return float(np.mean(self.metric_array("recall")))

    @property
    def mean_f1(self) -> float:
        return float(np.mean(self.metric_array("f1")))

    @property
    def mean_roc_auc(self) -> Optional[float]:
        vals = [f.roc_auc for f in self.folds if f.roc_auc is not None]
        return float(np.mean(vals)) if vals else None

    @property
    def std_roc_auc(self) -> Optional[float]:
        vals = [f.roc_auc for f in self.folds if f.roc_auc is not None]
        return float(np.std(vals)) if len(vals) >= 2 else None

    @property
    def best_fold(self) -> Optional[FoldMetrics]:
        if not self.folds:
            return None
        return max(self.folds, key=lambda f: f.accuracy)

    @property
    def worst_fold(self) -> Optional[FoldMetrics]:
        if not self.folds:
            return None
        return min(self.folds, key=lambda f: f.accuracy)


@dataclass
class LooImpactRecord:
    """
    Impacto de remover UNA muestra original del dataset de entrenamiento.

    Para una muestra dada `i` (señal original `signal_name`):
      - `acc_with`: accuracy de los modelos cuando esta muestra está
        en el train.
      - `acc_without`: accuracy de los modelos cuando esta muestra se
        excluye del train.
      - `delta`: acc_without − acc_with.
        · delta > 0  → quitar la muestra MEJORA el modelo (muestra
                       tóxica/etiqueta sospechosa).
        · delta ≈ 0  → la muestra es neutra.
        · delta < 0  → la muestra aporta señal útil al modelo.
    """
    signal_index: int          # índice 0..N-1 en la lista de señales originales
    signal_name: str
    class_label: int
    acc_with: float            # accuracy media sobre los modelos seleccionados
    acc_without: float
    delta: float               # acc_without - acc_with
    n_models: int


@dataclass
class CrossValidationResult:
    """Resultado completo de un run de validación cruzada."""
    config: CrossValidationConfig
    per_model: Dict[str, ModelCVResult] = field(default_factory=dict)

    # Identidad de cada muestra (después de augmentar) para correlacionarla
    # con el ranking de muestras conflictivas. `sample_origin[i]` es el
    # nombre legible (p. ej. "señal_001.csv [aug 3]") de la muestra i.
    sample_origin: List[str] = field(default_factory=list)
    # `y_true` global (no cambia entre modelos)
    y_true: Optional[np.ndarray] = None

    feature_names: List[str] = field(default_factory=list)
    total_time_s: float = 0.0
    timestamp: str = ""

    # Resultados específicos de LOO (vacío si la estrategia no fue 'loo')
    loo_impact: List[LooImpactRecord] = field(default_factory=list)
    loo_baseline_acc: Optional[float] = None    # accuracy media con TODAS las muestras


class CrossValidationWorker(QtCore.QObject):
    """
    Worker de validación cruzada. Soporta tres estrategias:
      - 'kfold'      : KFold simple
      - 'stratified' : StratifiedKFold (recomendado)
      - 'loo'        : Leave-One-Out por SEÑAL ORIGINAL. Por cada señal,
                       se entrena un modelo SIN ella y se compara la
                       accuracy frente al modelo completo. Identifica
                       muestras cuya presencia degrada generalización.
    """
    progress = QtCore.pyqtSignal(int, str)
    finished = QtCore.pyqtSignal(object)   # CrossValidationResult
    failed   = QtCore.pyqtSignal(str)
    log      = QtCore.pyqtSignal(str)      # mensaje de log

    def __init__(self, data_no: list, data_yes: list,
                 config: CrossValidationConfig):
        super().__init__()
        self.data_no, self.data_yes = data_no, data_yes
        self.config = config
        self._cancel = False

    @QtCore.pyqtSlot()
    def cancel(self):
        self._cancel = True

    @QtCore.pyqtSlot()
    def run(self):
        try:
            if self.config.strategy == "loo":
                self._run_loo()
            else:
                self._run_kfold()
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _run_kfold(self):
        try:
            t_global_start = time.time()
            self.log.emit(
                f"▶ Validación cruzada iniciada — "
                f"{self.config.strategy.upper()} con {self.config.n_splits} folds."
            )

            # 1) Extraer features con augmentación (mismo flujo que el
            #    entrenamiento real). Guardamos origen de cada muestra
            #    para poder rankear muestras conflictivas después.
            self.progress.emit(2, "Extrayendo features (clase Normal)…")
            feats_no, names_no = self._process_group_with_origin(
                self.data_no, label="Normal", offset=2, span=18
            )
            self.progress.emit(20, "Extrayendo features (clase Bypass)…")
            feats_yes, names_yes = self._process_group_with_origin(
                self.data_yes, label="Bypass", offset=20, span=18
            )

            if self._cancel: return

            if not feats_no or not feats_yes:
                raise ValueError(
                    "No se pudieron extraer features para ambas clases."
                )

            feature_names = sorted(feats_no[0].keys())
            X_no  = np.array([[f[k] for k in feature_names] for f in feats_no])
            X_yes = np.array([[f[k] for k in feature_names] for f in feats_yes])

            # Balanceo: replicar la clase minoritaria hasta target_total/2 cada una
            half = self.config.target_total // 2
            def resample(Xg, names, N):
                rng = np.random.default_rng(self.config.random_state)
                idx = rng.choice(len(Xg), size=N, replace=(len(Xg) < N))
                Xs = Xg[idx]
                ns = [names[j] for j in idx]
                return Xs, ns

            X_no_b,  names_no_b  = resample(X_no, names_no,  half)
            X_yes_b, names_yes_b = resample(X_yes, names_yes, self.config.target_total - half)
            X = np.vstack([X_no_b, X_yes_b])
            y = np.hstack([np.zeros(len(X_no_b), dtype=int),
                           np.ones(len(X_yes_b), dtype=int)])
            origins = names_no_b + names_yes_b

            # Permutación reproducible
            rng = np.random.default_rng(self.config.random_state)
            perm = rng.permutation(len(y))
            X, y, origins = X[perm], y[perm], [origins[i] for i in perm]
            self.log.emit(
                f"  Dataset final: {len(y)} muestras "
                f"({int(np.sum(y == 0))} Normal · {int(np.sum(y == 1))} Bypass)"
            )

            # 2) Seleccionar splitter
            if self.config.strategy == "stratified":
                splitter = StratifiedKFold(
                    n_splits=self.config.n_splits,
                    shuffle=self.config.shuffle,
                    random_state=self.config.random_state if self.config.shuffle else None,
                )
                self.log.emit("  Estrategia: StratifiedKFold (mantiene proporción de clases por fold)")
            else:
                splitter = KFold(
                    n_splits=self.config.n_splits,
                    shuffle=self.config.shuffle,
                    random_state=self.config.random_state if self.config.shuffle else None,
                )
                self.log.emit("  Estrategia: KFold (split aleatorio simple)")

            # Validar que los modelos están disponibles
            models_eff = [
                k for k in MODEL_KEYS
                if k in self.config.models_to_validate and model_is_available(k)
            ]
            if not models_eff:
                raise ValueError("Ningún modelo válido seleccionado.")
            self.log.emit(f"  Modelos a validar: {', '.join(models_eff)}")
            self.log.emit(f"  Calibración: {'activada' if self.config.calibrate else 'desactivada'}")

            # 3) Pre-calcular los índices de los folds una sola vez
            folds_indices = list(splitter.split(X, y))

            result = CrossValidationResult(
                config=self.config,
                per_model={},
                sample_origin=origins,
                y_true=y.copy(),
                feature_names=feature_names,
                timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
            )

            # 4) Loop por modelo × fold
            n_total = len(models_eff) * self.config.n_splits
            done = 0
            for m_idx, key in enumerate(models_eff):
                if self._cancel: return
                self.log.emit(f"\n▶ Modelo: {MODEL_DISPLAY_NAMES[key]}")
                mr = ModelCVResult(model_key=key)
                t_model_start = time.time()

                # Arrays para guardar predicciones por muestra original
                val_preds = np.full(len(y), -1, dtype=int)
                val_probs = np.full(len(y), np.nan, dtype=float)

                for fold_i, (tr_idx, va_idx) in enumerate(folds_indices):
                    if self._cancel: return
                    done += 1
                    pct = 35 + int(60 * done / n_total)
                    self.progress.emit(
                        pct,
                        f"{MODEL_SHORT_NAMES[key]} fold {fold_i+1}/{self.config.n_splits}…"
                    )

                    # Scaler dentro del fold para evitar leakage
                    scaler = StandardScaler().fit(X[tr_idx])
                    X_tr = scaler.transform(X[tr_idx])
                    X_va = scaler.transform(X[va_idx])
                    y_tr = y[tr_idx]; y_va = y[va_idx]

                    t_fold_start = time.time()
                    try:
                        base = _make_classifier(
                            key,
                            n_estimators=self.config.n_estimators,
                            svm_c=self.config.svm_c,
                            learning_rate=self.config.learning_rate,
                            max_depth=self.config.max_depth,
                        )
                        clf = _calibrated_wrap(
                            base, X_tr, y_tr,
                            enable=self.config.calibrate, cv=3
                        )
                        clf.fit(X_tr, y_tr)
                    except Exception as e:
                        self.log.emit(f"  Fold {fold_i+1}: error en fit — {e}")
                        continue
                    t_fit = time.time() - t_fold_start

                    yp = clf.predict(X_va)
                    pr = _prob_bypass(clf, X_va)

                    val_preds[va_idx] = yp
                    if pr is not None:
                        val_probs[va_idx] = pr

                    # Métricas
                    acc = float(accuracy_score(y_va, yp))
                    prec = float(precision_score(y_va, yp, zero_division=0))
                    rec  = float(recall_score(y_va, yp, zero_division=0))
                    f1   = float(f1_score(y_va, yp, zero_division=0))
                    auc: Optional[float] = None
                    if pr is not None and len(np.unique(y_va)) > 1:
                        try:
                            auc = float(roc_auc_score(y_va, pr))
                        except Exception:
                            auc = None
                    cm = confusion_matrix(y_va, yp, labels=[0, 1])
                    mr.folds.append(FoldMetrics(
                        fold_index=fold_i + 1,
                        accuracy=acc, precision=prec, recall=rec, f1=f1,
                        roc_auc=auc, confusion=cm,
                        train_time_s=t_fit,
                        n_train=len(tr_idx), n_val=len(va_idx),
                    ))
                    # Guardamos el modelo y scaler del fold para que la
                    # pestaña Ranking permita seleccionar/descargar este
                    # modelo concreto, no solo el "mejor de cada algoritmo".
                    mr.fold_models[fold_i + 1]  = clf
                    mr.fold_scalers[fold_i + 1] = scaler
                    self.log.emit(
                        f"  Fold {fold_i+1}: acc={acc:.3f} f1={f1:.3f} "
                        + (f"auc={auc:.3f} " if auc is not None else "")
                        + f"({t_fit:.2f}s)"
                    )

                mr.total_time_s = time.time() - t_model_start
                mr.val_preds = val_preds
                mr.val_probs = val_probs
                mr.val_truth = y.copy()
                result.per_model[key] = mr
                self.log.emit(
                    f"  Total {MODEL_SHORT_NAMES[key]}: "
                    f"acc_mean={mr.mean_accuracy:.4f} ± {mr.std_accuracy:.4f} "
                    f"({mr.total_time_s:.1f}s)"
                )

            result.total_time_s = time.time() - t_global_start
            self.progress.emit(100, "Validación cruzada completada.")
            self.log.emit(
                f"\n✅ CV completada en {result.total_time_s:.1f}s total."
            )
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())

    # ------------------------------------------------------------------
    # LEAVE-ONE-OUT POR SEÑAL ORIGINAL
    # ------------------------------------------------------------------
    def _run_loo(self):
        """
        Leave-One-Out por SEÑAL ORIGINAL (no por muestra augmentada).

        Estrategia:
            1. Calcula la accuracy de referencia con TODAS las señales
               originales (más sus augmentaciones).
            2. Para cada señal original i:
                 a) Construye el dataset SIN esa señal (ni sus augments).
                 b) Entrena los modelos seleccionados.
                 c) Evalúa la accuracy promedio sobre el mismo hold-out
                    interno (stratified split fijo, semilla fija para que
                    los resultados sean comparables).
                 d) Compara la accuracy "sin esa señal" vs "con esa señal".
            3. Las señales con `delta > 0` (rendimiento mejora al
               quitarlas) son sospechosas: probable etiqueta errónea,
               outlier o caso problemático.

        Diseño:
            - Para ahorrar tiempo, los modelos LOO usan los mismos
              hiperparámetros que el entrenamiento (vienen en config),
              pero idealmente con n_estimators reducido. El usuario
              puede bajar n_estimators a 50-100 antes de pulsar el
              botón para acelerar el análisis si tiene muchas señales.
            - La métrica reportada es accuracy en hold-out (no CV
              completa por muestra, porque sería N×N entrenamientos).
        """
        try:
            t_global_start = time.time()
            self.log.emit(
                "▶ Leave-One-Out por señal original."
            )

            # === 1) Extraer features (igual que en _run_kfold) ===========
            self.progress.emit(2, "Extrayendo features (clase Normal)…")
            feats_no, names_no, origin_no = self._process_group_with_signal_idx(
                self.data_no, label="Normal", class_label=0, offset=2, span=8
            )
            self.progress.emit(10, "Extrayendo features (clase Bypass)…")
            feats_yes, names_yes, origin_yes = self._process_group_with_signal_idx(
                self.data_yes, label="Bypass", class_label=1, offset=10, span=8
            )
            if self._cancel: return

            if not feats_no or not feats_yes:
                raise ValueError("No se pudieron extraer features para ambas clases.")

            feature_names = sorted(feats_no[0].keys())
            X_no  = np.array([[f[k] for k in feature_names] for f in feats_no])
            X_yes = np.array([[f[k] for k in feature_names] for f in feats_yes])

            # `sig_idx_arr[i]` = índice global de la señal original que generó
            # la muestra i (0..n_signals_total-1). Lo necesitamos para
            # poder excluir TODAS las augmentaciones de una señal a la vez.
            X = np.vstack([X_no, X_yes])
            y = np.hstack([
                np.zeros(len(X_no), dtype=int),
                np.ones(len(X_yes), dtype=int),
            ])
            # En LOO, cada (signal_idx, class_label) identifica una señal
            # original única. Mapeamos a un índice global continuo:
            #   señales Normal:  0..len(data_no)-1
            #   señales Bypass: len(data_no)..len(data_no)+len(data_yes)-1
            all_origins = origin_no + origin_yes
            sig_idx_arr = np.array(
                [o["sig_idx"] for o in all_origins], dtype=int
            )
            sig_names = [None] * (len(self.data_no) + len(self.data_yes))
            sig_classes = [None] * (len(self.data_no) + len(self.data_yes))
            for i, sig in enumerate(self.data_no):
                sig_names[i] = os.path.basename(sig[0])
                sig_classes[i] = 0
            for i, sig in enumerate(self.data_yes):
                sig_names[len(self.data_no) + i] = os.path.basename(sig[0])
                sig_classes[len(self.data_no) + i] = 1

            n_signals = len(self.data_no) + len(self.data_yes)
            self.log.emit(
                f"  {n_signals} señales originales · "
                f"{len(X)} muestras totales (con augmentación)"
            )

            # === 2) Modelos a evaluar ===
            models_eff = [
                k for k in MODEL_KEYS
                if k in self.config.models_to_validate and model_is_available(k)
            ]
            if not models_eff:
                raise ValueError("Ningún modelo válido seleccionado.")
            self.log.emit(f"  Modelos: {', '.join(models_eff)}")

            # === 3) Hold-out interno fijo para evaluación comparable ===
            # Hacemos UN stratified split de los datos como referencia
            # común. La accuracy reportada en cada iteración LOO es
            # accuracy en este test set fijo. La idea: si quito una señal
            # del TRAIN, ¿cambia el rendimiento en el test? El test no
            # cambia entre iteraciones.
            from sklearn.model_selection import train_test_split as _tts
            try:
                idx_tr_full, idx_te = _tts(
                    np.arange(len(X)), test_size=0.25,
                    random_state=self.config.random_state,
                    stratify=y,
                )
            except Exception:
                idx_tr_full = np.arange(len(X))
                idx_te = np.arange(len(X))
            X_te = X[idx_te]; y_te = y[idx_te]
            sig_idx_test = sig_idx_arr[idx_te]   # señales presentes en test
            self.log.emit(
                f"  Hold-out interno: {len(idx_tr_full)} train · "
                f"{len(idx_te)} test"
            )

            def fit_and_score(idx_train: np.ndarray) -> Dict[str, float]:
                """Entrena los modelos en `idx_train` y devuelve acc por modelo en el test fijo."""
                scaler = StandardScaler().fit(X[idx_train])
                Xtr = scaler.transform(X[idx_train])
                Xte = scaler.transform(X_te)
                ytr = y[idx_train]
                accs: Dict[str, float] = {}
                for key in models_eff:
                    try:
                        base = _make_classifier(
                            key,
                            n_estimators=self.config.n_estimators,
                            svm_c=self.config.svm_c,
                            learning_rate=self.config.learning_rate,
                            max_depth=self.config.max_depth,
                        )
                        clf = _calibrated_wrap(
                            base, Xtr, ytr,
                            enable=self.config.calibrate, cv=3
                        )
                        clf.fit(Xtr, ytr)
                        yp = clf.predict(Xte)
                        accs[key] = float(accuracy_score(y_te, yp))
                    except Exception:
                        accs[key] = float("nan")
                return accs

            # === 4) Baseline: accuracy con TODAS las señales ===
            self.progress.emit(20, "Calculando baseline con todas las señales…")
            t_baseline = time.time()
            baseline_accs = fit_and_score(idx_tr_full)
            self.log.emit(
                f"  Baseline ({time.time() - t_baseline:.2f}s): "
                + " | ".join(f"{k}={baseline_accs[k]:.4f}" for k in models_eff)
            )
            baseline_mean = float(np.nanmean(list(baseline_accs.values())))

            # === 5) Loop por señal original ===
            loo_records: List[LooImpactRecord] = []

            # Para cada modelo, también tracking del baseline por modelo
            per_model: Dict[str, ModelCVResult] = {}
            for key in models_eff:
                mr = ModelCVResult(model_key=key)
                # Guardamos un único "fold" sintético con la accuracy baseline
                mr.folds.append(FoldMetrics(
                    fold_index=0, accuracy=baseline_accs[key],
                    precision=0.0, recall=0.0, f1=0.0, roc_auc=None,
                    confusion=np.zeros((2, 2), dtype=int),
                    train_time_s=0.0,
                    n_train=len(idx_tr_full), n_val=len(idx_te),
                ))
                per_model[key] = mr

            for i in range(n_signals):
                if self._cancel: return
                pct = 25 + int(70 * (i + 1) / n_signals)
                self.progress.emit(
                    pct,
                    f"LOO {i+1}/{n_signals}: {sig_names[i]}"
                )

                # Excluir TODAS las muestras (incluidas augmentaciones)
                # de la señal i del set de train.
                mask_keep = (sig_idx_arr[idx_tr_full] != i)
                idx_train_loo = idx_tr_full[mask_keep]

                # Edge case: si la señal i fue toda al test (raro), no hay
                # nada que excluir → mismo dataset que baseline.
                if len(idx_train_loo) < 2 or len(np.unique(y[idx_train_loo])) < 2:
                    # No se puede entrenar sin esta señal — la marcamos
                    # como neutra (delta=0) para no romper el ranking.
                    loo_records.append(LooImpactRecord(
                        signal_index=i, signal_name=sig_names[i],
                        class_label=sig_classes[i],
                        acc_with=baseline_mean, acc_without=baseline_mean,
                        delta=0.0, n_models=len(models_eff),
                    ))
                    continue

                accs_without = fit_and_score(idx_train_loo)
                acc_without_mean = float(np.nanmean(list(accs_without.values())))
                delta = acc_without_mean - baseline_mean

                loo_records.append(LooImpactRecord(
                    signal_index=i, signal_name=sig_names[i],
                    class_label=sig_classes[i],
                    acc_with=baseline_mean,
                    acc_without=acc_without_mean,
                    delta=delta,
                    n_models=len(models_eff),
                ))

                # Log resumido (no cada señal, demasiado verbose)
                if (i + 1) % max(1, n_signals // 10) == 0 or i == n_signals - 1:
                    sign = "+" if delta > 0 else ""
                    self.log.emit(
                        f"  [{i+1}/{n_signals}] {sig_names[i]}: "
                        f"acc_sin={acc_without_mean:.4f}  Δ={sign}{delta:.4f}"
                    )

            # === 6) Empaquetar resultado ===
            result = CrossValidationResult(
                config=self.config,
                per_model=per_model,
                sample_origin=[],   # no aplica para LOO
                y_true=None,
                feature_names=feature_names,
                timestamp=datetime.datetime.now().isoformat(timespec="seconds"),
                loo_impact=loo_records,
                loo_baseline_acc=baseline_mean,
            )
            result.total_time_s = time.time() - t_global_start
            self.progress.emit(100, "Leave-One-Out completado.")

            # Resumen al log
            toxic = sorted(loo_records, key=lambda r: -r.delta)
            n_toxic = sum(1 for r in toxic if r.delta > 0)
            self.log.emit(
                f"\n✅ LOO completado en {result.total_time_s:.1f}s · "
                f"baseline={baseline_mean:.4f} · "
                f"{n_toxic} señal(es) con delta > 0 (sospechosas)"
            )
            if toxic[:5]:
                self.log.emit("\n  Top 5 más tóxicas (mayor mejora al quitarlas):")
                for r in toxic[:5]:
                    if r.delta <= 0: break
                    self.log.emit(
                        f"    {r.signal_name} (clase {r.class_label}) "
                        f"Δ=+{r.delta:.4f}"
                    )
            self.finished.emit(result)
        except Exception:
            self.failed.emit(traceback.format_exc())

    def _process_group_with_signal_idx(self, group, label: str,
                                         class_label: int,
                                         offset: int, span: int):
        """
        Variante de `_process_group_with_origin` que además devuelve un
        dict de metadatos por muestra con la clave 'sig_idx', usado por
        el flujo LOO para agrupar augmentaciones por señal original.

        El índice 'sig_idx' es GLOBAL: Normal=0..len(data_no)-1,
        Bypass=len(data_no)..len(data_no)+len(data_yes)-1.
        """
        feats = []
        names = []
        origins = []
        total = max(1, len(group))
        n_data_no = len(self.data_no)
        for i, (path, t, p, fs) in enumerate(group):
            base_name = os.path.basename(path)
            sig_idx = i if class_label == 0 else (n_data_no + i)
            for j, (t2, p2) in enumerate(
                augment_single_signal(t, p, n_aug=self.config.n_aug)
            ):
                try:
                    feats.append(extract_features(t2, p2, fs))
                    names.append(
                        base_name if j == 0 else f"{base_name} [aug {j}]"
                    )
                    origins.append({"sig_idx": sig_idx, "aug_idx": j})
                except Exception:
                    pass
            self.progress.emit(
                offset + int(span * (i + 1) / total),
                f"[{label}] {base_name}",
            )
        return feats, names, origins

    def _process_group_with_origin(self, group, label: str,
                                    offset: int, span: int):
        """
        Igual que `_process_group` del entrenador, pero también devuelve
        una lista de nombres legibles por cada feature-vector — uno por
        cada muestra augmentada.
        """
        feats = []
        names = []
        total = max(1, len(group))
        for i, (path, t, p, fs) in enumerate(group):
            base_name = os.path.basename(path)
            for j, (t2, p2) in enumerate(
                augment_single_signal(t, p, n_aug=self.config.n_aug)
            ):
                try:
                    feats.append(extract_features(t2, p2, fs))
                    names.append(
                        base_name if j == 0 else f"{base_name} [aug {j}]"
                    )
                except Exception:
                    pass
            self.progress.emit(
                offset + int(span * (i + 1) / total),
                f"[{label}] procesando… {base_name}",
            )
        return feats, names


# ============================================================================
# 7. WIDGETS REUTILIZABLES
# ============================================================================

def _apply_ax_theme(ax):
    ax.set_facecolor(COLOR_PANEL_ALT)
    ax.tick_params(colors=COLOR_TEXT_DIM, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color(COLOR_BORDER)
    ax.grid(True, linestyle="--", alpha=0.25, color=COLOR_BORDER)
    ax.title.set_color(COLOR_TEXT)
    ax.xaxis.label.set_color(COLOR_TEXT_DIM)
    ax.yaxis.label.set_color(COLOR_TEXT_DIM)


class _NumericTableItem(QtWidgets.QTableWidgetItem):
    """
    Item de QTableWidget que muestra texto pero ordena por un valor
    numérico almacenado aparte. Definido a nivel de módulo (no como
    clase anidada dentro de un método) para que la operación de
    comparación al ordenar sea estable a través de redibujos sucesivos
    de la tabla — definirla cada vez creaba clases distintas y eso
    provocaba inestabilidad en el sort entre re-evaluaciones.
    """
    def __init__(self, text: str, value: float):
        super().__init__(text)
        self._val = float(value)

    def __lt__(self, other):
        try:
            return self._val < other._val
        except Exception:
            return super().__lt__(other)


class PhysicalRangePanel(QtWidgets.QGroupBox):
    """
    Panel reutilizable con controles para los rangos físicos de
    generación de transientes sintéticos.

    Para cada parámetro físico (p0, A, f0, tau, t0, noise_std) expone
    dos `QDoubleSpinBox` (min y max) que el usuario puede ajustar.
    El método `get_ranges()` devuelve un dict listo para pasar a
    `generate_dataset(param_ranges=...)`.

    Llamando a `set_ranges(ranges)` se rellenan los controles con los
    valores que se le pasen.

    Los defaults provienen de DEFAULT_PARAM_RANGES.
    """
    # Definición central de los controles: clave, etiqueta, sufijo,
    # decimales, paso, rango UI.
    _SPECS = [
        ("p0",        "Presión base p₀",   "bar", 2, 0.1,  (0.1, 20.0)),
        ("A",         "Amplitud A",        "",    2, 0.05, (0.0, 5.0)),
        ("f0",        "Frecuencia f₀",     "Hz",  1, 1.0,  (1.0, 500.0)),
        ("tau",       "Decaimiento τ",     "s",   3, 0.01, (0.001, 5.0)),
        ("t0",        "Inicio t₀",         "s",   2, 0.05, (0.0, 10.0)),
        ("noise_std", "Ruido σ",           "",    4, 0.001,(0.0, 0.5)),
    ]

    def __init__(self, title: str = "Parámetros físicos del transiente",
                 parent: Optional[QtWidgets.QWidget] = None):
        super().__init__(title, parent)
        form = QtWidgets.QFormLayout(self)
        form.setVerticalSpacing(6)

        self._mins: Dict[str, QtWidgets.QDoubleSpinBox] = {}
        self._maxs: Dict[str, QtWidgets.QDoubleSpinBox] = {}

        info = QtWidgets.QLabel(
            "Cada parámetro se muestrea uniformemente entre [min, max]\n"
            "para cada señal generada. Defaults reflejan la práctica\n"
            "habitual; ajusta si quieres concentrar el dataset en\n"
            "escenarios específicos."
        )
        info.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:8pt; font-style:italic;")
        info.setWordWrap(True)
        form.addRow(info)

        for key, label, suffix, dec, step, ui_range in self._SPECS:
            default_lo, default_hi = DEFAULT_PARAM_RANGES[key]

            spin_min = QtWidgets.QDoubleSpinBox()
            spin_min.setRange(*ui_range); spin_min.setSingleStep(step)
            spin_min.setDecimals(dec); spin_min.setValue(default_lo)
            if suffix:
                spin_min.setSuffix(f" {suffix}")
            spin_max = QtWidgets.QDoubleSpinBox()
            spin_max.setRange(*ui_range); spin_max.setSingleStep(step)
            spin_max.setDecimals(dec); spin_max.setValue(default_hi)
            if suffix:
                spin_max.setSuffix(f" {suffix}")

            self._mins[key] = spin_min
            self._maxs[key] = spin_max

            row = QtWidgets.QWidget()
            h = QtWidgets.QHBoxLayout(row)
            h.setContentsMargins(0, 0, 0, 0); h.setSpacing(4)
            h.addWidget(spin_min, 1)
            sep = QtWidgets.QLabel("→"); sep.setStyleSheet(f"color:{COLOR_TEXT_DIM};")
            h.addWidget(sep)
            h.addWidget(spin_max, 1)
            form.addRow(f"{label}:", row)

        # Botón pequeño de reset al fondo
        btn_reset = QtWidgets.QPushButton("↺  Restaurar valores por defecto")
        btn_reset.setToolTip("Restaura los rangos físicos a sus valores por defecto.")
        btn_reset.clicked.connect(self.reset_defaults)
        form.addRow(btn_reset)

    # ------------------------------------------------------------------
    def get_ranges(self) -> Dict[str, Tuple[float, float]]:
        """Devuelve los rangos actuales como dict listo para generate_dataset."""
        out: Dict[str, Tuple[float, float]] = {}
        for key in self._mins:
            lo = float(self._mins[key].value())
            hi = float(self._maxs[key].value())
            if hi < lo:  # tolerar inversión por edición
                lo, hi = hi, lo
            out[key] = (lo, hi)
        return out

    def set_ranges(self, ranges: Dict[str, Tuple[float, float]]):
        for key, (lo, hi) in ranges.items():
            if key in self._mins:
                self._mins[key].setValue(float(lo))
                self._maxs[key].setValue(float(hi))

    def reset_defaults(self):
        self.set_ranges(DEFAULT_PARAM_RANGES)


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, nrows=1, ncols=1):
        self.fig = Figure(facecolor=COLOR_PANEL, tight_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)
        self.nrows, self.ncols = nrows, ncols
        self.axes = self.fig.subplots(nrows, ncols, squeeze=False)
        for row in self.axes:
            for ax in row:
                _apply_ax_theme(ax)

    def clear_axes(self):
        for row in self.axes:
            for ax in row:
                ax.clear()
                _apply_ax_theme(ax)

    def reset_figure(self):
        """
        Limpia COMPLETAMENTE la figura y reconstruye los `axes` originales.

        A diferencia de `clear_axes()`, este método elimina también los
        ejes del colorbar (y cualquier otro eje extra que se haya añadido
        a la figura). Es lo correcto cuando se va a redibujar un plot
        que incluye un colorbar — si no, los colorbars previos se apilan
        a la derecha y comprimen el área principal en cada redibujo.
        """
        self.fig.clear()
        self.axes = self.fig.subplots(self.nrows, self.ncols, squeeze=False)
        for row in self.axes:
            for ax in row:
                _apply_ax_theme(ax)

    def show_empty(self, msg="Sin datos"):
        self.clear_axes()
        for row in self.axes:
            for ax in row:
                ax.text(0.5, 0.5, msg, color=COLOR_TEXT_DIM, ha="center",
                        va="center", fontsize=11, transform=ax.transAxes, alpha=0.6)
        self.draw_idle()


class CollapsibleSection(QtWidgets.QWidget):
    """
    Sección plegable con cabecera clicable.

    Modo simple (`activatable=False`):
        Una cabecera con flecha ▶/▼ que expande/colapsa el cuerpo.

    Modo activable (`activatable=True`):
        La cabecera incluye un CHECKBOX a la izquierda que indica si la
        sección está "activa" (verde) o "inactiva" (azul).
        - Click en el checkbox → activa/desactiva (emite `activated`).
        - Click en el resto de la cabecera → expande/colapsa.

    Esto permite usarlo como contenedor para un sub-filtro: el checkbox
    es el "Activar (xxx)", y el cuerpo plegable contiene los parámetros.
    """

    toggled   = QtCore.pyqtSignal(bool)  # expandir/colapsar
    activated = QtCore.pyqtSignal(bool)  # solo en modo activable

    def __init__(self, title: str, parent=None, start_open: bool = False,
                 activatable: bool = False, start_active: bool = False):
        super().__init__(parent)
        self._title = title
        self._is_open = bool(start_open)
        self._activatable = bool(activatable)
        self._is_active = bool(start_active)

        # ── Construcción de la cabecera ─────────────────────────
        self.header = QtWidgets.QFrame(self)
        self.header.setObjectName("collapsibleHeader")
        self.header.setCursor(QtCore.Qt.PointingHandCursor)
        self.header.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Fixed)
        self.header.installEventFilter(self)  # capturar clicks fuera del checkbox

        h_lay = QtWidgets.QHBoxLayout(self.header)
        h_lay.setContentsMargins(8, 5, 10, 5)
        h_lay.setSpacing(8)

        # Checkbox solo en modo activable
        if self._activatable:
            self.activate_check = QtWidgets.QCheckBox()
            self.activate_check.setChecked(self._is_active)
            self.activate_check.setCursor(QtCore.Qt.PointingHandCursor)
            self.activate_check.toggled.connect(self._on_activate_toggled)
            h_lay.addWidget(self.activate_check)
        else:
            self.activate_check = None

        # Etiqueta con flecha + título
        self.lbl_arrow = QtWidgets.QLabel(self._arrow_char())
        self.lbl_arrow.setStyleSheet(f"color:{COLOR_ACCENT}; font-weight:700; font-size:10pt;")
        h_lay.addWidget(self.lbl_arrow)

        self.lbl_title = QtWidgets.QLabel(self._title)
        self.lbl_title.setStyleSheet(f"font-weight:700; font-size:10pt;")
        h_lay.addWidget(self.lbl_title)
        h_lay.addStretch(1)

        # ── Cuerpo plegable ─────────────────────────────────────
        self.body = QtWidgets.QWidget(self)
        self.body.setObjectName("collapsibleBody")
        self.body.setVisible(self._is_open)

        # Layout vertical principal
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)
        lay.addWidget(self.header)
        lay.addWidget(self.body)

        self._apply_styles()

    # ------------------------------------------------------------------
    def _arrow_char(self) -> str:
        return "▼" if self._is_open else "▶"

    def _apply_styles(self):
        """Recolorea cabecera y cuerpo según estado activo/inactivo."""
        # Color de cabecera: verde si activa, azul si no, gris si no es activable
        if self._activatable:
            if self._is_active:
                head_bg     = "#2a3a2e"  # verde oscuro
                head_border = COLOR_SUCCESS
                title_color = COLOR_SUCCESS
                body_border = COLOR_SUCCESS
            else:
                head_bg     = COLOR_PANEL_ALT
                head_border = COLOR_BORDER
                title_color = COLOR_TEXT_DIM
                body_border = COLOR_BORDER
        else:
            head_bg     = COLOR_PANEL_ALT
            head_border = COLOR_BORDER
            title_color = COLOR_ACCENT
            body_border = COLOR_BORDER

        # Bordes ajustados según estado abierto/cerrado
        if self._is_open:
            head_radius = "border-top-left-radius:4px; border-top-right-radius:4px; border-bottom-left-radius:0; border-bottom-right-radius:0;"
            head_border_bottom = "border-bottom: 1px solid " + body_border + ";"
        else:
            head_radius = "border-radius:4px;"
            head_border_bottom = ""

        self.header.setStyleSheet(f"""
            QFrame#collapsibleHeader {{
                background-color: {head_bg};
                border: 1px solid {head_border};
                {head_radius}
                {head_border_bottom}
            }}
            QFrame#collapsibleHeader:hover {{
                border-color: {COLOR_ACCENT};
            }}
        """)
        self.lbl_arrow.setText(self._arrow_char())
        self.lbl_arrow.setStyleSheet(f"color:{title_color}; font-weight:700; font-size:10pt;")
        self.lbl_title.setStyleSheet(f"color:{title_color}; font-weight:700; font-size:10pt;")
        self.body.setStyleSheet(f"""
            QWidget#collapsibleBody {{
                background-color: {COLOR_PANEL};
                border: 1px solid {body_border};
                border-top: none;
                border-bottom-left-radius: 4px;
                border-bottom-right-radius: 4px;
            }}
        """)

    # ------------------------------------------------------------------
    def eventFilter(self, obj, event):
        """Capturar click sobre el header (excluyendo el área del checkbox)."""
        if obj is self.header and event.type() == QtCore.QEvent.MouseButtonRelease:
            # Si el click cayó dentro del rectángulo del checkbox, lo ignoramos
            # (el checkbox lo maneja por sí mismo).
            if self.activate_check is not None:
                cb_rect = self.activate_check.geometry()
                if cb_rect.contains(event.pos()):
                    return False
            self._toggle_open()
            return True
        return super().eventFilter(obj, event)

    def _toggle_open(self):
        self._is_open = not self._is_open
        self.body.setVisible(self._is_open)
        self._apply_styles()
        self.toggled.emit(self._is_open)

    def _on_activate_toggled(self, checked: bool):
        self._is_active = bool(checked)
        self._apply_styles()
        self.activated.emit(self._is_active)

    # ------------------------------------------------------------------
    # API pública
    def setContentLayout(self, layout: QtWidgets.QLayout):
        self.body.setLayout(layout)

    def setOpen(self, open_state: bool):
        if open_state == self._is_open:
            return
        self._toggle_open()

    def isOpen(self) -> bool:
        return self._is_open

    def setActive(self, active: bool):
        if self.activate_check is None:
            return
        self.activate_check.setChecked(bool(active))

    def isActive(self) -> bool:
        return self._is_active


class PredictionBadge(QtWidgets.QFrame):
    """
    Badge de predicción para el simulador.

    Diseño: un veredicto principal arriba (NORMAL / BYPASS / sin
    predicción) con su % de confianza, seguido de un botón colapsable
    «▼ Ver detalle» que despliega un panel con el % por cada modelo
    individual.

    Cuando hay 2+ modelos cargados, el veredicto principal proviene del
    voting ensemble (soft voting). Cuando hay un solo modelo cargado,
    el veredicto principal viene directamente de ese modelo.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("predBadge")
        self.setMinimumHeight(90)
        lay = QtWidgets.QVBoxLayout(self)
        lay.setContentsMargins(15, 10, 15, 10)
        lay.setSpacing(4)

        self.title = QtWidgets.QLabel("Sin predicción")
        self.title.setObjectName("badgeTitle")
        self.title.setAlignment(QtCore.Qt.AlignCenter)
        lay.addWidget(self.title)

        self.detail = QtWidgets.QLabel("Entrena o carga un modelo, luego predice.")
        self.detail.setObjectName("badgeDetail")
        self.detail.setAlignment(QtCore.Qt.AlignCenter)
        self.detail.setWordWrap(True)
        lay.addWidget(self.detail)

        # Toggle "Ver detalle" — solo se muestra si hay ≥2 modelos
        # cuyo desglose individual aporte algo nuevo respecto al
        # veredicto principal.
        self.toggle_btn = QtWidgets.QToolButton()
        self.toggle_btn.setObjectName("badgeToggle")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(False)
        self.toggle_btn.setText("▼  Ver detalle por modelo")
        self.toggle_btn.setCursor(QtCore.Qt.PointingHandCursor)
        self.toggle_btn.setToolButtonStyle(QtCore.Qt.ToolButtonTextOnly)
        self.toggle_btn.toggled.connect(self._on_toggle)
        self.toggle_btn.setVisible(False)
        lay.addWidget(self.toggle_btn, alignment=QtCore.Qt.AlignCenter)

        # Panel con detalle por modelo (oculto por defecto)
        self.detail_panel = QtWidgets.QFrame()
        self.detail_panel.setObjectName("badgeDetailPanel")
        self._detail_layout = QtWidgets.QVBoxLayout(self.detail_panel)
        self._detail_layout.setContentsMargins(10, 6, 10, 6)
        self._detail_layout.setSpacing(3)
        self.detail_panel.setVisible(False)
        lay.addWidget(self.detail_panel)

        self.set_neutral()

    # ------------------------------------------------------------------
    def _style(self, border, bg, text_color):
        self.setStyleSheet(f"""
            QFrame#predBadge {{
                background-color: {bg};
                border: 2px solid {border};
                border-radius: 8px;
            }}
            QLabel#badgeTitle {{
                color: {text_color}; font-size: 16pt; font-weight: 700;
                background: transparent; border: none;
            }}
            QLabel#badgeDetail {{
                color: {COLOR_TEXT}; font-size: 9pt;
                background: transparent; border: none;
            }}
            QToolButton#badgeToggle {{
                color: {text_color};
                background: transparent; border: none;
                font-size: 9pt; font-weight: 600;
                padding: 2px 8px;
            }}
            QToolButton#badgeToggle:hover {{
                color: {COLOR_TEXT};
            }}
            QFrame#badgeDetailPanel {{
                background-color: rgba(0, 0, 0, 60);
                border: 1px solid {border};
                border-radius: 4px;
            }}
        """)

    def _clear_detail_panel(self):
        """Borra todas las filas del panel detalle."""
        while self._detail_layout.count() > 0:
            item = self._detail_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _on_toggle(self, checked: bool):
        if checked:
            self.toggle_btn.setText("▲  Ocultar detalle")
            self.detail_panel.setVisible(True)
        else:
            self.toggle_btn.setText("▼  Ver detalle por modelo")
            self.detail_panel.setVisible(False)

    # ------------------------------------------------------------------
    def set_neutral(self):
        self.title.setText("⏸  Sin predicción")
        self.detail.setText("Entrena o carga un modelo, luego predice.")
        self._clear_detail_panel()
        self.toggle_btn.setVisible(False)
        self.detail_panel.setVisible(False)
        self.toggle_btn.setChecked(False)
        self._style(COLOR_BORDER, COLOR_PANEL_ALT, COLOR_TEXT_DIM)

    # ------------------------------------------------------------------
    # Métodos legacy (mantenidos por retrocompat con código antiguo)
    # ------------------------------------------------------------------
    def set_normal(self, conf_rf, conf_svm):
        per = {}
        if conf_rf  is not None: per["rf"]  = {"pred": 0, "conf": conf_rf,  "short": "RF"}
        if conf_svm is not None: per["svm"] = {"pred": 0, "conf": conf_svm, "short": "SVM"}
        self.set_decision(0, max(filter(None, [conf_rf, conf_svm]), default=1.0), per)

    def set_bypass(self, conf_rf, conf_svm):
        per = {}
        if conf_rf  is not None: per["rf"]  = {"pred": 1, "conf": conf_rf,  "short": "RF"}
        if conf_svm is not None: per["svm"] = {"pred": 1, "conf": conf_svm, "short": "SVM"}
        self.set_decision(1, max(filter(None, [conf_rf, conf_svm]), default=1.0), per)

    def set_disagree(self, pred_rf, pred_svm, conf_rf, conf_svm):
        per = {
            "rf":  {"pred": pred_rf,  "conf": conf_rf  or 0, "short": "RF"},
            "svm": {"pred": pred_svm, "conf": conf_svm or 0, "short": "SVM"},
        }
        # Sin ensemble explícito, usamos voto mayoritario simple
        avg = ((conf_rf or 0) + (conf_svm or 0)) / 2
        self.set_decision(pred_rf if conf_rf and conf_rf >= (conf_svm or 0) else pred_svm,
                          avg, per)

    # ------------------------------------------------------------------
    # API principal nueva
    # ------------------------------------------------------------------
    def set_predictions(self, per_model: Dict[str, Dict[str, Any]]):
        """API antigua, conservada por compat. Usa set_decision detrás."""
        if not per_model:
            self.set_neutral(); return
        # Si hay ensemble, usamos su predicción como veredicto.
        if "ensemble" in per_model:
            ens = per_model["ensemble"]
            individual = {k: v for k, v in per_model.items() if k != "ensemble"}
            self.set_decision(ens["pred"], ens["conf"], individual,
                               source_label="Voting Ensemble")
        else:
            # Un solo modelo
            (k, info), = list(per_model.items())[:1] if len(per_model) == 1 \
                          else (list(per_model.items())[:1])
            # Caso general: tomar el primero como veredicto
            first_key, first_info = next(iter(per_model.items()))
            self.set_decision(first_info["pred"], first_info["conf"],
                               per_model,
                               source_label=MODEL_DISPLAY_NAMES.get(first_key, first_key))

    def set_decision(self, pred: int, conf: float,
                       per_model: Dict[str, Dict[str, Any]],
                       source_label: Optional[str] = None):
        """
        Establece el veredicto principal del badge.

        Parameters
        ----------
        pred : 0 (Normal) o 1 (Bypass) — la decisión a mostrar arriba.
        conf : confianza de la decisión, en [0, 1]. Es la probabilidad
               de la clase predicha (no la prob_bypass cruda).
        per_model : dict de modelos individuales (sin ensemble) para el
                    panel desplegable. Cada valor: {pred, conf, short}.
        source_label : texto que indica de dónde viene la decisión (ej.
                       "Voting Ensemble" si hay 2+ modelos, o el nombre
                       del modelo si solo hay uno).
        """
        labels = {0: "Normal", 1: "Bypass"}

        # Texto principal
        if pred == 0:
            self.title.setText("✅  SISTEMA NORMAL")
            border, bg, fg = COLOR_SUCCESS, "#1e3320", COLOR_SUCCESS
        else:
            self.title.setText("⚠️  ¡BYPASS DETECTADO!")
            border, bg, fg = COLOR_DANGER, "#3a1a22", COLOR_DANGER

        # Línea inferior: % + fuente
        src_txt = f" — según {source_label}" if source_label else ""
        self.detail.setText(
            f"{conf*100:.1f}%  de confianza en {labels[pred]}{src_txt}"
        )
        self._style(border, bg, fg)

        # Construir el panel detallado
        self._clear_detail_panel()
        # Solo mostramos el toggle si hay al menos 2 modelos individuales
        # (con un solo modelo, su info ya está en la línea principal).
        n_models = len(per_model)
        if n_models >= 2:
            for key, info in per_model.items():
                if key == "ensemble":  # nunca debería estar aquí
                    continue
                row = self._make_detail_row(
                    info.get("short", key.upper()),
                    info["pred"], info["conf"], pred
                )
                self._detail_layout.addWidget(row)
            self.toggle_btn.setVisible(True)
            # Conservar estado abierto/cerrado entre predicciones
            self.detail_panel.setVisible(self.toggle_btn.isChecked())
        else:
            self.toggle_btn.setVisible(False)
            self.detail_panel.setVisible(False)

        # v4.0: el veredicto entra deslizándose con fade + glow del color
        # del resultado. El glow se aplica DESPUÉS de la animación de
        # opacidad (un widget solo soporta un QGraphicsEffect a la vez).
        FX.slide_fade_in(self, dy=14, duration=FX.DURATION_MED)
        QtCore.QTimer.singleShot(
            FX.DURATION_MED + 30,
            lambda b=border: FX.glow(self, b, blur=24, alpha=110)
        )

    def _make_detail_row(self, short_name: str, pred: int, conf: float,
                          final_pred: int) -> QtWidgets.QWidget:
        """
        Crea una fila del panel de detalle:
            [SHORT]  [Normal/Bypass]  [██████░░░░ 65.3%]  [✓ o ✗]
        El check/aspa indica si la predicción de este modelo coincide
        con el veredicto final.
        """
        labels = {0: "Normal", 1: "Bypass"}
        agrees = (pred == final_pred)

        row = QtWidgets.QWidget()
        h = QtWidgets.QHBoxLayout(row)
        h.setContentsMargins(0, 0, 0, 0); h.setSpacing(8)

        # Nombre del modelo (RF, SVM, XGB, LGBM)
        lbl_name = QtWidgets.QLabel(f"<b>{short_name}</b>")
        lbl_name.setMinimumWidth(46)
        lbl_name.setStyleSheet(f"color:{COLOR_TEXT}; background:transparent;")
        h.addWidget(lbl_name)

        # Predicción del modelo
        pred_color = COLOR_DANGER if pred == 1 else COLOR_SUCCESS
        lbl_pred = QtWidgets.QLabel(labels[pred])
        lbl_pred.setMinimumWidth(60)
        lbl_pred.setStyleSheet(
            f"color:{pred_color}; font-weight:600; background:transparent;"
        )
        h.addWidget(lbl_pred)

        # Barra de confianza + porcentaje
        bar_container = QtWidgets.QWidget()
        bar_lay = QtWidgets.QHBoxLayout(bar_container)
        bar_lay.setContentsMargins(0, 0, 0, 0); bar_lay.setSpacing(6)
        bar = QtWidgets.QProgressBar()
        bar.setRange(0, 100)
        bar.setValue(int(round(conf * 100)))
        bar.setTextVisible(False)
        bar.setFixedHeight(10)
        # Color de la barra según predicción
        bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: rgba(255,255,255,30);
                border: 1px solid rgba(255,255,255,40);
                border-radius: 5px;
            }}
            QProgressBar::chunk {{
                background-color: {pred_color};
                border-radius: 4px;
            }}
        """)
        bar_lay.addWidget(bar, 1)
        lbl_pct = QtWidgets.QLabel(f"{conf*100:.1f}%")
        lbl_pct.setMinimumWidth(50)
        lbl_pct.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        lbl_pct.setStyleSheet(f"color:{COLOR_TEXT}; background:transparent;")
        bar_lay.addWidget(lbl_pct)
        h.addWidget(bar_container, 1)

        # Marcador de coincidencia con el veredicto final
        check = QtWidgets.QLabel("✓" if agrees else "✗")
        check.setMinimumWidth(16)
        check.setAlignment(QtCore.Qt.AlignCenter)
        check.setToolTip(
            "Coincide con el veredicto final" if agrees
            else "Discrepa del veredicto final"
        )
        check.setStyleSheet(
            f"color:{COLOR_SUCCESS if agrees else COLOR_WARNING};"
            "font-weight:700; background:transparent;"
        )
        h.addWidget(check)

        return row


# ============================================================================
# 8. PESTAÑAS
# ============================================================================

class SimulatorTab(QtWidgets.QWidget):
    """Pestaña 1 — Generar / cargar una señal, visualizarla, filtrarla y predecir."""

    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self.current_t:      Optional[np.ndarray] = None
        self.current_p:      Optional[np.ndarray] = None    # señal cruda
        self.current_p_filt: Optional[np.ndarray] = None    # señal filtrada
        self.current_fs:     Optional[int] = None
        self.current_diag:   Dict[str, np.ndarray] = {}     # diagnóstico filtro
        self.filter_config = FilterConfig()
        self._last_label: str = "Señal"
        self._last_loaded_path: Optional[str] = None  # para sugerir nombre al guardar
        self._build_ui()

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(10)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_left())
        splitter.addWidget(self._build_center())
        splitter.addWidget(self._build_right())
        splitter.setSizes([360, 640, 360])
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

    def _build_left(self):
        # Panel izquierdo con scroll (ahora tiene más controles por el filtrado)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)

        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 6, 0)

        # ─── Parámetros de simulación (sección plegable) ───────────
        grp = CollapsibleSection("⚙  Parámetros de simulación", start_open=False)
        param_body = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(param_body)
        form.setVerticalSpacing(8)
        form.setContentsMargins(10, 8, 10, 10)

        def spin(cls, mn, mx, val, step=1, suf="", dec=None, tip=""):
            s = cls(); s.setRange(mn, mx)
            if dec is not None: s.setDecimals(dec)
            s.setSingleStep(step); s.setValue(val)
            if suf: s.setSuffix(suf)
            if tip: s.setToolTip(tip)
            return s

        self.duration = spin(QtWidgets.QDoubleSpinBox, 1.0, 20.0, 5.0, 0.5, " s", 2,
                             "Duración total")
        self.fs       = spin(QtWidgets.QSpinBox,       200, 5000, 2000, 100, " Hz", None,
                             "Frecuencia de muestreo")
        self.p0       = spin(QtWidgets.QDoubleSpinBox, 0.1, 10.0, 2.5, 0.1, " bar", 2,
                             "Presión base")
        self.A        = spin(QtWidgets.QDoubleSpinBox, 0.0, 2.0, 0.6, 0.05, "", 2,
                             "Amplitud del transiente")
        self.f0       = spin(QtWidgets.QDoubleSpinBox, 1.0, 500.0, 25.0, 1.0, " Hz", 1,
                             "Frecuencia dominante")
        self.tau      = spin(QtWidgets.QDoubleSpinBox, 0.01, 5.0, 0.4, 0.05, " s", 2,
                             "Constante de decaimiento")
        self.t0       = spin(QtWidgets.QDoubleSpinBox, 0.0, 2.0, 0.5, 0.05, " s", 2,
                             "Inicio del transiente")
        self.noise    = spin(QtWidgets.QDoubleSpinBox, 0.0, 0.1, 0.01, 0.001, "", 3,
                             "Ruido (σ)")
        self.bypass   = QtWidgets.QCheckBox("Simular bypass clandestino")
        self.bypass.setToolTip("Atenúa amplitud, reduce τ, desplaza f₀.")

        for lbl, w_ in [
            ("Duración:", self.duration), ("Muestreo (fs):", self.fs),
            ("Presión base p₀:", self.p0), ("Amplitud A:", self.A),
            ("Frecuencia f₀:", self.f0), ("Decaimiento τ:", self.tau),
            ("Inicio t₀:", self.t0), ("Ruido σ:", self.noise),
            ("", self.bypass),
        ]:
            form.addRow(lbl, w_)

        # Botón de generar dentro de la sección plegable
        btn_sim = QtWidgets.QPushButton("▶  Generar simulación")
        btn_sim.setObjectName("primaryButton")
        btn_sim.setMinimumHeight(36)
        btn_sim.setToolTip(
            "Genera una señal sintética con los parámetros de arriba.\n"
            "Se mostrará en la gráfica y se podrá analizar/predecir."
        )
        btn_sim.clicked.connect(self.on_simulate)
        form.addRow(btn_sim)

        # Inyectar el body en el CollapsibleSection
        grp.setContentLayout(QtWidgets.QVBoxLayout())
        grp.body.layout().setContentsMargins(0, 0, 0, 0)
        grp.body.layout().setSpacing(0)
        grp.body.layout().addWidget(param_body)
        lay.addWidget(grp)

        # ─── CSV ───────────────────────────────────────────────────
        grp2 = QtWidgets.QGroupBox("📂  CSV")
        gl = QtWidgets.QHBoxLayout(grp2)
        btn_load = QtWidgets.QPushButton("⬆  Cargar")
        btn_load.setToolTip(
            "Cargar una señal desde un archivo de texto.\n\n"
            "Formatos aceptados: .csv, .txt, .log, .dat, .tsv\n"
            "Separadores válidos: coma, punto y coma, tabulación, espacios.\n\n"
            "El parser es tolerante: ignora cabeceras del programa\n"
            "(p. ej. «PuTTY log…»), cabeceras de columna (p. ej.\n"
            "«time_s,pressure_bar»), líneas vacías y marcadores finales\n"
            "(p. ej. «FIN»). Solo conserva las líneas con dos números."
        )
        btn_save = QtWidgets.QPushButton("⬇  Guardar")
        btn_save.setToolTip(
            "Guardar la señal en un archivo CSV.\n\n"
            "• Si «Activar filtrado» está APAGADO  → guarda la señal cruda.\n"
            "• Si «Activar filtrado» está ENCENDIDO → guarda la señal filtrada\n"
            "  (con todas las etapas activas aplicadas)."
        )
        btn_load.clicked.connect(self.on_load_csv)
        btn_save.clicked.connect(self.on_save_csv)
        gl.addWidget(btn_load); gl.addWidget(btn_save)
        lay.addWidget(grp2)

        # ─── Filtrado de señal (sección plegable, apagado por defecto) ──
        grp_filt = CollapsibleSection("🪒  Filtrado de señal", start_open=False)
        # El body del grupo es un QWidget con un VBoxLayout que contiene
        # el master, los botones de sugerencia y las subsecciones por filtro.
        body = QtWidgets.QWidget()
        body_lay = QtWidgets.QVBoxLayout(body)
        body_lay.setContentsMargins(10, 8, 10, 10)
        body_lay.setSpacing(8)

        # — Master toggle ─────────────────────────────────────────
        master_row = QtWidgets.QFrame()
        master_row.setObjectName("filterMasterRow")
        mr_lay = QtWidgets.QHBoxLayout(master_row)
        mr_lay.setContentsMargins(8, 6, 8, 6); mr_lay.setSpacing(8)
        self.filter_enable = QtWidgets.QCheckBox("Activar filtrado")
        self.filter_enable.setChecked(False)
        self.filter_enable.setToolTip(
            "Activa el pipeline completo de filtrado.\n"
            "Limpia ruido impulsivo del sensor (Arduino, etc.) preservando\n"
            "el transiente del golpe de ariete.\n\n"
            "Está APAGADO por defecto: actívalo solo cuando lo necesites."
        )
        self.filter_enable.setStyleSheet("font-weight:700;")
        mr_lay.addWidget(self.filter_enable)
        mr_lay.addStretch(1)
        master_row.setStyleSheet(
            f"QFrame#filterMasterRow {{"
            f"  background-color: {COLOR_PANEL_ALT};"
            f"  border: 1px solid {COLOR_BORDER};"
            f"  border-radius: 4px; }}"
        )
        body_lay.addWidget(master_row)

        # — Sugerencias automáticas ───────────────────────────────
        sugg_row = QtWidgets.QHBoxLayout()
        sugg_row.setSpacing(6)
        self.btn_suggest = QtWidgets.QPushButton("🪄  Activar valores sugeridos")
        self.btn_suggest.setToolTip(
            "Analiza la señal cargada y sugiere valores adaptados para\n"
            "todos los parámetros del pipeline (ventanas, umbrales, etc.).\n"
            "El sistema mide la frecuencia de muestreo, el ruido típico\n"
            "y la densidad de spikes, y rellena los controles con\n"
            "parámetros que suelen funcionar bien para esa señal.\n\n"
            "Requiere haber cargado o generado una señal primero."
        )
        self.btn_suggest.clicked.connect(self._on_suggest_filter_params)
        self.btn_reset_filter = QtWidgets.QPushButton("↺  Restaurar valores por defecto")
        self.btn_reset_filter.setToolTip(
            "Vuelve los parámetros a sus valores por defecto originales.\n"
            "No cambia el estado activo/inactivo de cada filtro."
        )
        self.btn_reset_filter.clicked.connect(self._on_reset_filter_params)
        sugg_row.addWidget(self.btn_suggest)
        sugg_row.addWidget(self.btn_reset_filter)
        body_lay.addLayout(sugg_row)

        # === SUBSECCIÓN 1) Diferencia con vecinos ================
        sec1 = CollapsibleSection("1) Diferencia con vecinos",
                                   start_open=False, activatable=True,
                                   start_active=False)
        b1 = QtWidgets.QWidget()
        f1 = QtWidgets.QFormLayout(b1)
        f1.setContentsMargins(10, 8, 10, 10); f1.setVerticalSpacing(6)

        info1 = QtWidgets.QLabel(
            "Anti-spike de 1 muestra. Compara cada valor con el promedio\n"
            "de sus vecinos x[i−1] y x[i+1]; si difiere mucho y los\n"
            "vecinos están de acuerdo entre sí, lo trata como spike.\n"
            "Ejemplo: 3.00 → 8.00 → 3.12 ⇒ 8.00 se reemplaza por 3.06."
        )
        info1.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:8pt; font-style:italic;")
        info1.setWordWrap(True)
        f1.addRow(info1)

        self.neighbor_sigmas = QtWidgets.QDoubleSpinBox()
        self.neighbor_sigmas.setRange(1.0, 20.0); self.neighbor_sigmas.setSingleStep(0.5)
        self.neighbor_sigmas.setValue(4.0); self.neighbor_sigmas.setDecimals(1)
        self.neighbor_sigmas.setToolTip(
            "Umbral en múltiplos del ruido típico sample-a-sample.\n"
            "Más bajo = más agresivo. 4.0 es conservador."
        )
        self.neighbor_passes = QtWidgets.QSpinBox()
        self.neighbor_passes.setRange(1, 5); self.neighbor_passes.setValue(2)
        self.neighbor_passes.setToolTip(
            "Pasadas iterativas. 2 es suficiente para spikes de 1-2 muestras."
        )
        f1.addRow("Vecinos n·σ:",     self.neighbor_sigmas)
        f1.addRow("Vecinos pasadas:", self.neighbor_passes)
        sec1.setContentLayout(QtWidgets.QVBoxLayout())
        sec1.body.layout().setContentsMargins(0, 0, 0, 0); sec1.body.layout().addWidget(b1)
        body_lay.addWidget(sec1)
        self.sec_neighbor = sec1

        # === SUBSECCIÓN 2) Hampel =================================
        sec2 = CollapsibleSection("2) Hampel (mediana móvil)",
                                   start_open=False, activatable=True,
                                   start_active=False)
        b2 = QtWidgets.QWidget(); f2 = QtWidgets.QFormLayout(b2)
        f2.setContentsMargins(10, 8, 10, 10); f2.setVerticalSpacing(6)
        info2 = QtWidgets.QLabel(
            "Mediana móvil + MAD sobre una ventana corta. Captura spikes\n"
            "de 2-3 muestras consecutivas que escapan al filtro de vecinos."
        )
        info2.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:8pt; font-style:italic;")
        info2.setWordWrap(True)
        f2.addRow(info2)
        self.hampel_window = QtWidgets.QSpinBox()
        self.hampel_window.setRange(3, 51); self.hampel_window.setSingleStep(2)
        self.hampel_window.setValue(7)
        self.hampel_window.setToolTip("Tamaño de ventana en muestras (impar).")
        self.hampel_sigmas = QtWidgets.QDoubleSpinBox()
        self.hampel_sigmas.setRange(1.0, 10.0); self.hampel_sigmas.setSingleStep(0.5)
        self.hampel_sigmas.setValue(3.0); self.hampel_sigmas.setDecimals(1)
        self.hampel_sigmas.setToolTip(
            "Umbral de detección (n·σ). 3.0 conservador; 2.5 más agresivo."
        )
        f2.addRow("Hampel ventana:", self.hampel_window)
        f2.addRow("Hampel n·σ:",     self.hampel_sigmas)
        sec2.setContentLayout(QtWidgets.QVBoxLayout())
        sec2.body.layout().setContentsMargins(0, 0, 0, 0); sec2.body.layout().addWidget(b2)
        body_lay.addWidget(sec2)
        self.sec_hampel = sec2

        # === SUBSECCIÓN 3) Envolvente IQR =========================
        sec3 = CollapsibleSection("3) Envolvente IQR (racimos densos)",
                                   start_open=False, activatable=True,
                                   start_active=False)
        b3 = QtWidgets.QWidget(); f3 = QtWidgets.QFormLayout(b3)
        f3.setContentsMargins(10, 8, 10, 10); f3.setVerticalSpacing(6)
        info3 = QtWidgets.QLabel(
            "Suprime racimos densos de 4-7 spikes consecutivos que escapan\n"
            "a Hampel. Usa percentiles Q1/Q3 sobre una ventana ancha.\n"
            "Tiene protección automática de transientes reales."
        )
        info3.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:8pt; font-style:italic;")
        info3.setWordWrap(True)
        f3.addRow(info3)
        self.iqr_window = QtWidgets.QSpinBox()
        self.iqr_window.setRange(11, 201); self.iqr_window.setSingleStep(2)
        self.iqr_window.setValue(31)
        self.iqr_window.setToolTip(
            "Ventana del rolling-IQR (impar). 31 es óptima para 200-500 Hz."
        )
        self.iqr_k = QtWidgets.QDoubleSpinBox()
        self.iqr_k.setRange(1.0, 10.0); self.iqr_k.setSingleStep(0.5)
        self.iqr_k.setValue(3.0); self.iqr_k.setDecimals(1)
        self.iqr_k.setToolTip(
            "Multiplicador del IQR. Outlier si fuera de [Q1−k·IQR, Q3+k·IQR]."
        )
        self.iqr_passes = QtWidgets.QSpinBox()
        self.iqr_passes.setRange(1, 10); self.iqr_passes.setValue(3)
        self.iqr_passes.setToolTip("Pasadas iterativas. 3 suele bastar.")
        self.iqr_protect = QtWidgets.QCheckBox("Proteger transientes reales")
        self.iqr_protect.setChecked(True)
        self.iqr_protect.setToolTip(
            "Detecta zonas de cambio sostenido (cierres reales) y NO\n"
            "aplica el filtro en ellas — distingue cierre real de racimos."
        )
        f3.addRow("IQR ventana:", self.iqr_window)
        f3.addRow("IQR k·IQR:",   self.iqr_k)
        f3.addRow("IQR pasadas:", self.iqr_passes)
        f3.addRow("",             self.iqr_protect)
        sec3.setContentLayout(QtWidgets.QVBoxLayout())
        sec3.body.layout().setContentsMargins(0, 0, 0, 0); sec3.body.layout().addWidget(b3)
        body_lay.addWidget(sec3)
        self.sec_iqr = sec3

        # === SUBSECCIÓN 4) Filtro de duración =====================
        sec4 = CollapsibleSection("4) Filtro de duración (spike vs transiente real)",
                                   start_open=False, activatable=True,
                                   start_active=False)
        b4 = QtWidgets.QWidget(); f4 = QtWidgets.QFormLayout(b4)
        f4.setContentsMargins(10, 8, 10, 10); f4.setVerticalSpacing(6)
        info4 = QtWidgets.QLabel(
            "Distingue spikes (cortos) de transientes reales (largos)\n"
            "por la duración:\n"
            "  • <50 ms = SPIKE → eliminar\n"
            "  • ≥50 ms = TRANSIENTE REAL → preservar"
        )
        info4.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:8pt; font-style:italic;")
        info4.setWordWrap(True)
        f4.addRow(info4)
        self.dur_baseline = QtWidgets.QSpinBox()
        self.dur_baseline.setRange(101, 1001); self.dur_baseline.setSingleStep(50)
        self.dur_baseline.setValue(401)
        self.dur_baseline.setToolTip("Ventana de la base. 401 ≈ 1.2 s a 333 Hz.")
        self.dur_k = QtWidgets.QDoubleSpinBox()
        self.dur_k.setRange(1.5, 8.0); self.dur_k.setSingleStep(0.25)
        self.dur_k.setValue(3.5); self.dur_k.setDecimals(2)
        self.dur_k.setToolTip("Umbral en sigmas robustos. 3.5 = óptimo.")
        self.dur_min = QtWidgets.QDoubleSpinBox()
        self.dur_min.setRange(0.005, 1.0); self.dur_min.setSingleStep(0.01)
        self.dur_min.setValue(0.05); self.dur_min.setDecimals(3)
        self.dur_min.setSuffix(" s")
        self.dur_min.setToolTip("Anomalías más cortas = spike. 0.050 s recomendado.")
        self.dur_passes = QtWidgets.QSpinBox()
        self.dur_passes.setRange(1, 10); self.dur_passes.setValue(3)
        self.dur_passes.setToolTip("Pasadas iterativas.")
        f4.addRow("Dur. ventana:",  self.dur_baseline)
        f4.addRow("Dur. n·σ:",      self.dur_k)
        f4.addRow("Dur. mín:",      self.dur_min)
        f4.addRow("Dur. pasadas:",  self.dur_passes)
        sec4.setContentLayout(QtWidgets.QVBoxLayout())
        sec4.body.layout().setContentsMargins(0, 0, 0, 0); sec4.body.layout().addWidget(b4)
        body_lay.addWidget(sec4)
        self.sec_duration = sec4

        # === SUBSECCIÓN 5) Eliminación manual por intervalo ======
        sec5 = CollapsibleSection("5) Eliminación manual por intervalo",
                                   start_open=False, activatable=True,
                                   start_active=False)
        b5 = QtWidgets.QWidget(); f5 = QtWidgets.QFormLayout(b5)
        f5.setContentsMargins(10, 8, 10, 10); f5.setVerticalSpacing(6)
        info5 = QtWidgets.QLabel(
            "Elimina quirúrgicamente picos identificados visualmente.\n"
            "Define [t_inicio, t_fin] con un umbral; el filtro elimina los\n"
            "picos que cumplan el criterio dentro de cada intervalo."
        )
        info5.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:8pt; font-style:italic;")
        info5.setWordWrap(True)
        f5.addRow(info5)

        # Tabla de intervalos
        self.manual_table = QtWidgets.QTableWidget(0, 5)
        self.manual_table.setHorizontalHeaderLabels(
            ["✓", "t inicio (s)", "t fin (s)", "umbral (bar)", "modo"]
        )
        self.manual_table.setEditTriggers(
            QtWidgets.QAbstractItemView.DoubleClicked |
            QtWidgets.QAbstractItemView.EditKeyPressed |
            QtWidgets.QAbstractItemView.AnyKeyPressed
        )
        self.manual_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )
        self.manual_table.setSelectionMode(
            QtWidgets.QAbstractItemView.SingleSelection
        )
        self.manual_table.setMinimumHeight(110)
        self.manual_table.setMaximumHeight(180)
        self.manual_table.verticalHeader().setVisible(False)
        hdr = self.manual_table.horizontalHeader()
        hdr.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)
        hdr.setSectionResizeMode(4, QtWidgets.QHeaderView.ResizeToContents)
        self.manual_table.setToolTip(
            "Cada fila = un intervalo. Doble click sobre una celda para editar.\n"
            "  ✓: activar/desactivar el intervalo\n"
            "  t inicio, t fin: rango de tiempo en segundos\n"
            "  umbral: nivel de presión en bar\n"
            "  modo: '>' elimina picos por encima, '<' por debajo"
        )
        self.manual_table.itemChanged.connect(self._on_manual_changed)
        f5.addRow(self.manual_table)

        man_btn_row = QtWidgets.QWidget()
        man_btn_lay = QtWidgets.QHBoxLayout(man_btn_row)
        man_btn_lay.setContentsMargins(0, 0, 0, 0); man_btn_lay.setSpacing(6)
        self.btn_manual_add = QtWidgets.QPushButton("+ Agregar intervalo")
        self.btn_manual_add.setToolTip(
            "Agrega un intervalo nuevo a la lista. Edítalo con doble click."
        )
        self.btn_manual_add.clicked.connect(self._on_manual_add)
        self.btn_manual_del = QtWidgets.QPushButton("− Quitar selección")
        self.btn_manual_del.setToolTip("Elimina la fila seleccionada (o la última).")
        self.btn_manual_del.clicked.connect(self._on_manual_del)
        self.btn_manual_clear = QtWidgets.QPushButton("Vaciar")
        self.btn_manual_clear.setToolTip("Elimina todos los intervalos.")
        self.btn_manual_clear.clicked.connect(self._on_manual_clear)
        man_btn_lay.addWidget(self.btn_manual_add)
        man_btn_lay.addWidget(self.btn_manual_del)
        man_btn_lay.addWidget(self.btn_manual_clear)
        f5.addRow(man_btn_row)
        sec5.setContentLayout(QtWidgets.QVBoxLayout())
        sec5.body.layout().setContentsMargins(0, 0, 0, 0); sec5.body.layout().addWidget(b5)
        body_lay.addWidget(sec5)
        self.sec_manual = sec5

        # === SUBSECCIÓN 6) Pasa-bajos Butterworth ================
        sec6 = CollapsibleSection("6) Pasa-bajos Butterworth",
                                   start_open=False, activatable=True,
                                   start_active=False)
        b6 = QtWidgets.QWidget(); f6 = QtWidgets.QFormLayout(b6)
        f6.setContentsMargins(10, 8, 10, 10); f6.setVerticalSpacing(6)
        info6 = QtWidgets.QLabel(
            "Atenúa ruido de alta frecuencia residual sin desfase\n"
            "(filtfilt, fase cero)."
        )
        info6.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:8pt; font-style:italic;")
        info6.setWordWrap(True)
        f6.addRow(info6)
        self.lp_cutoff = QtWidgets.QDoubleSpinBox()
        self.lp_cutoff.setRange(10.0, 5000.0); self.lp_cutoff.setSingleStep(10.0)
        self.lp_cutoff.setValue(150.0); self.lp_cutoff.setSuffix(" Hz")
        self.lp_cutoff.setToolTip(
            "Frecuencia de corte. Por encima del contenido útil (≈10-100 Hz)\n"
            "y por debajo de Nyquist (fs/2)."
        )
        f6.addRow("LP cutoff:", self.lp_cutoff)
        sec6.setContentLayout(QtWidgets.QVBoxLayout())
        sec6.body.layout().setContentsMargins(0, 0, 0, 0); sec6.body.layout().addWidget(b6)
        body_lay.addWidget(sec6)
        self.sec_lowpass = sec6

        # === Visualización ========================================
        sec_vis = CollapsibleSection("Visualización",
                                      start_open=False, activatable=False)
        bv = QtWidgets.QWidget(); fv = QtWidgets.QFormLayout(bv)
        fv.setContentsMargins(10, 8, 10, 10); fv.setVerticalSpacing(6)
        self.show_raw_overlay = QtWidgets.QCheckBox("Superponer señal cruda")
        self.show_raw_overlay.setChecked(True)
        self.show_raw_overlay.setToolTip(
            "Muestra la señal cruda en gris translúcido sobre la filtrada,\n"
            "con marcadores de cada etapa:\n"
            "  +  naranja  → diferencia con vecinos\n"
            "  ×  rojo     → Hampel\n"
            "  ★  magenta  → envolvente IQR (racimo)\n"
            "  ▼  amarillo → filtro de duración\n"
            "  ◆  cian     → eliminación manual"
        )
        fv.addRow("", self.show_raw_overlay)
        sec_vis.setContentLayout(QtWidgets.QVBoxLayout())
        sec_vis.body.layout().setContentsMargins(0, 0, 0, 0); sec_vis.body.layout().addWidget(bv)
        body_lay.addWidget(sec_vis)

        # Inyectar `body` en el grupo principal
        grp_filt.setContentLayout(QtWidgets.QVBoxLayout())
        grp_filt.body.layout().setContentsMargins(0, 0, 0, 0)
        grp_filt.body.layout().setSpacing(0)
        grp_filt.body.layout().addWidget(body)
        lay.addWidget(grp_filt)

        # ── Variables proxy para que el resto del código siga
        # ── usando self.filter_neighbor.isChecked() etc. sin cambios ─
        # Cada subsección expone su checkbox interno como "filter_xxx"
        # para mantener compatibilidad con el código existente.
        self.filter_neighbor = sec1.activate_check
        self.filter_hampel   = sec2.activate_check
        self.filter_iqr      = sec3.activate_check
        self.filter_duration = sec4.activate_check
        self.filter_manual   = sec5.activate_check
        self.filter_lp       = sec6.activate_check

        # ── Conexiones para reaplicar el filtro ante cualquier cambio
        for cb in (self.filter_enable, self.filter_neighbor,
                   self.filter_hampel, self.filter_iqr, self.iqr_protect,
                   self.filter_duration, self.filter_manual,
                   self.filter_lp, self.show_raw_overlay):
            cb.toggled.connect(self._on_filter_changed)
        for sp in (self.neighbor_sigmas, self.neighbor_passes,
                   self.hampel_window, self.hampel_sigmas,
                   self.iqr_window, self.iqr_k, self.iqr_passes,
                   self.dur_baseline, self.dur_k, self.dur_min, self.dur_passes,
                   self.lp_cutoff):
            sp.valueChanged.connect(self._on_filter_changed)

        lay.addStretch()
        scroll.setWidget(w)
        return scroll

    def _build_center(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0); lay.setSpacing(4)

        self.viz_tabs = QtWidgets.QTabWidget()
        self.viz_tabs.setDocumentMode(True)

        # Tiempo + PSD
        self.c_signal = PlotCanvas(nrows=2, ncols=1)
        t1 = QtWidgets.QWidget(); t1l = QtWidgets.QVBoxLayout(t1)
        t1l.setContentsMargins(0, 0, 0, 0)
        t1l.addWidget(NavigationToolbar(self.c_signal, self))
        t1l.addWidget(self.c_signal)
        self.viz_tabs.addTab(t1, "📈  Señal & PSD")

        # Espectrograma
        self.c_spec = PlotCanvas(nrows=1, ncols=1)
        t2 = QtWidgets.QWidget(); t2l = QtWidgets.QVBoxLayout(t2)
        t2l.setContentsMargins(0, 0, 0, 0)
        t2l.addWidget(NavigationToolbar(self.c_spec, self))
        t2l.addWidget(self.c_spec)
        self.viz_tabs.addTab(t2, "🌈  Espectrograma")

        # Wavelet
        self.c_wave = PlotCanvas(nrows=1, ncols=1)
        t3 = QtWidgets.QWidget(); t3l = QtWidgets.QVBoxLayout(t3)
        t3l.setContentsMargins(0, 0, 0, 0)
        t3l.addWidget(NavigationToolbar(self.c_wave, self))
        t3l.addWidget(self.c_wave)
        self.viz_tabs.addTab(t3, "🌊  Wavelet")

        # ── v4.0: Vista 3D del espectrograma ──────────────────────────
        # Render PEREZOSO: la superficie 3D solo se calcula cuando el
        # usuario abre esta pestaña (y la señal cambió desde la última
        # vez). Así la generación/carga de señales sigue siendo
        # instantánea aunque el plot 3D tarde unos cientos de ms.
        self.c_3d = Plot3DCanvas()
        t4 = QtWidgets.QWidget(); t4l = QtWidgets.QVBoxLayout(t4)
        t4l.setContentsMargins(0, 0, 0, 0)
        hint3d = QtWidgets.QLabel(
            "  Arrastra con el mouse para rotar la superficie · "
            "rueda para zoom"
        )
        hint3d.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:8pt;")
        # (v4.2) FIX layout fullscreen: el QLabel tenía política vertical
        # Preferred (crecible), así que en pantalla completa absorbía el
        # espacio sobrante del layout y empujaba el canvas 3D al fondo
        # dejando un hueco enorme arriba. Lo fijamos en altura y damos
        # TODO el stretch al canvas.
        hint3d.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                             QtWidgets.QSizePolicy.Fixed)
        t4l.addWidget(hint3d)
        t4l.addWidget(self.c_3d, 1)   # stretch=1 → el canvas llena el resto
        self.viz_tabs.addTab(t4, "🧊  Vista 3D")
        self._3d_dirty = False          # hay señal nueva sin renderizar
        self._3d_payload = None         # (t, p, fs, baseline, suffix)
        self.viz_tabs.currentChanged.connect(self._maybe_render_3d)

        lay.addWidget(self.viz_tabs)
        self.c_signal.show_empty("Genera o carga una señal")
        self.c_spec.show_empty()
        self.c_wave.show_empty()
        self.c_3d.show_empty()
        return w

    def _maybe_render_3d(self, idx: int = -1):
        """Renderiza la vista 3D solo si su pestaña está activa y hay
        datos pendientes (render perezoso)."""
        if not self._3d_dirty or self._3d_payload is None:
            return
        current = self.viz_tabs.currentWidget()
        # ¿La pestaña actual contiene el canvas 3D?
        if current is None or self.c_3d not in current.findChildren(Plot3DCanvas):
            return
        t, p, fs, baseline, suffix = self._3d_payload
        try:
            self.c_3d.plot_spectrogram_surface(t, p, fs, baseline, suffix)
        except Exception as e:
            self.c_3d.show_empty(f"Error en vista 3D: {e}")
        self._3d_dirty = False

    def _build_right(self):
        w = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        grp = QtWidgets.QGroupBox("🔮  Resultado de predicción")
        gl = QtWidgets.QVBoxLayout(grp)
        self.badge = PredictionBadge()
        gl.addWidget(self.badge)
        btn_pred = QtWidgets.QPushButton("🎯  Predecir escenario actual")
        btn_pred.setObjectName("accentButton")
        btn_pred.setMinimumHeight(38)
        btn_pred.setCursor(QtCore.Qt.PointingHandCursor)
        btn_pred.clicked.connect(self.on_predict)
        gl.addWidget(btn_pred)
        # v4.0: glow «respirando» — es el call-to-action de la pestaña
        FX.pulse_glow(btn_pred, COLOR_SUCCESS, blur_min=6, blur_max=26)
        lay.addWidget(grp)

        grp2 = QtWidgets.QGroupBox("📊  Características extraídas")
        gl2 = QtWidgets.QVBoxLayout(grp2)
        self.features_text = QtWidgets.QPlainTextEdit()
        self.features_text.setReadOnly(True)
        gl2.addWidget(self.features_text)
        lay.addWidget(grp2, 1)
        return w

    # ---------- helpers de filtrado ----------
    def _build_filter_config_from_ui(self) -> FilterConfig:
        return FilterConfig(
            enabled=self.filter_enable.isChecked(),
            neighbor_enabled=self.filter_neighbor.isChecked(),
            neighbor_threshold_sigmas=self.neighbor_sigmas.value(),
            neighbor_max_passes=self.neighbor_passes.value(),
            neighbor_agree_ratio=0.5,  # default sensato; no expuesto en UI
            hampel_enabled=self.filter_hampel.isChecked(),
            hampel_window=self.hampel_window.value() | 1,  # asegurar impar
            hampel_n_sigmas=self.hampel_sigmas.value(),
            iqr_enabled=self.filter_iqr.isChecked(),
            iqr_window=self.iqr_window.value() | 1,
            iqr_k=self.iqr_k.value(),
            iqr_max_passes=self.iqr_passes.value(),
            iqr_protect_transients=self.iqr_protect.isChecked(),
            iqr_protect_window=201,  # fijo: cubre ~0.4-2.2 s a 100-500 Hz
            duration_enabled=self.filter_duration.isChecked(),
            duration_baseline_window=self.dur_baseline.value() | 1,
            duration_k_sigmas=self.dur_k.value(),
            duration_min_transient_s=self.dur_min.value(),
            duration_max_passes=self.dur_passes.value(),
            manual_enabled=self.filter_manual.isChecked(),
            manual_intervals=self._read_manual_intervals(),
            lowpass_enabled=self.filter_lp.isChecked(),
            lowpass_cutoff=self.lp_cutoff.value(),
        )

    # ---------- gestión de la tabla de intervalos manuales ----------
    def _read_manual_intervals(self) -> List[ManualInterval]:
        """Construye la lista de ManualInterval desde la tabla."""
        out: List[ManualInterval] = []
        for row in range(self.manual_table.rowCount()):
            try:
                # Col 0 = checkbox de "enabled"
                chk_item = self.manual_table.item(row, 0)
                enabled = (chk_item is not None
                           and chk_item.checkState() == QtCore.Qt.Checked)
                # Cols 1, 2, 3 son números
                t_start = float(self.manual_table.item(row, 1).text())
                t_end   = float(self.manual_table.item(row, 2).text())
                thr     = float(self.manual_table.item(row, 3).text())
                # Col 4 es un combo
                combo = self.manual_table.cellWidget(row, 4)
                mode = combo.currentText() if combo is not None else ">"
                out.append(ManualInterval(t_start=t_start, t_end=t_end,
                                          threshold=thr, mode=mode,
                                          enabled=enabled))
            except (ValueError, AttributeError, TypeError):
                # Fila incompleta o mal formateada: ignorar
                continue
        return out

    def _on_manual_add(self):
        """Agrega una fila nueva con valores razonables a la tabla manual."""
        # Por defecto, usar el rango completo de la señal cargada
        if self.current_t is not None and len(self.current_t) > 0:
            t0 = float(self.current_t[0])
            t1 = float(self.current_t[-1])
            # Usar un tramo central por defecto (10% del total)
            span = (t1 - t0) * 0.1
            mid  = 0.5 * (t0 + t1)
            t_start_def = max(t0, mid - span / 2)
            t_end_def   = min(t1, mid + span / 2)
            # Umbral por defecto: percentil 95 de la señal cruda
            thr_def = float(np.percentile(self.current_p, 95))
        else:
            t_start_def, t_end_def, thr_def = 0.0, 1.0, 5.0

        # Bloquear señales para no disparar reapply mientras populamos la fila
        self.manual_table.blockSignals(True)
        row = self.manual_table.rowCount()
        self.manual_table.insertRow(row)

        # Col 0: checkbox (item con check state)
        chk = QtWidgets.QTableWidgetItem()
        chk.setFlags(chk.flags() | QtCore.Qt.ItemIsUserCheckable)
        chk.setCheckState(QtCore.Qt.Checked)
        chk.setTextAlignment(QtCore.Qt.AlignCenter)
        self.manual_table.setItem(row, 0, chk)

        # Cols 1-3: editables numéricamente
        for col, val in [(1, t_start_def), (2, t_end_def), (3, thr_def)]:
            it = QtWidgets.QTableWidgetItem(f"{val:.3f}")
            it.setTextAlignment(QtCore.Qt.AlignCenter)
            self.manual_table.setItem(row, col, it)

        # Col 4: combo
        cmb = QtWidgets.QComboBox()
        cmb.addItems([">", "<"])
        cmb.setCurrentIndex(0)
        cmb.currentTextChanged.connect(lambda *_: self._on_filter_changed())
        self.manual_table.setCellWidget(row, 4, cmb)

        self.manual_table.blockSignals(False)
        self._on_filter_changed()

    def _on_manual_del(self):
        """Elimina la fila seleccionada (o la última si no hay selección)."""
        n = self.manual_table.rowCount()
        if n == 0:
            return
        sel = self.manual_table.currentRow()
        if sel < 0:
            sel = n - 1  # quitar la última si no hay selección
        self.manual_table.removeRow(sel)
        self._on_filter_changed()

    def _on_manual_clear(self):
        """Vacía toda la tabla."""
        if self.manual_table.rowCount() == 0:
            return
        self.manual_table.blockSignals(True)
        self.manual_table.setRowCount(0)
        self.manual_table.blockSignals(False)
        self._on_filter_changed()

    def _on_manual_changed(self, item):
        """Se dispara al editar una celda. Re-aplica el filtro."""
        # Validación numérica suave: si la celda no parsea, restaurar valor anterior.
        if item.column() in (1, 2, 3):
            try:
                float(item.text())
            except ValueError:
                self.manual_table.blockSignals(True)
                item.setText("0.000")
                self.manual_table.blockSignals(False)
        self._on_filter_changed()

    # ---------- sugerencias automáticas / valores por defecto ----------
    def _block_filter_signals(self, block: bool):
        """Bloquea/desbloquea las señales de los controles del filtro
        para que cambiar varios a la vez solo dispare un único reapply
        al final."""
        targets = [
            self.filter_enable, self.filter_neighbor, self.filter_hampel,
            self.filter_iqr, self.iqr_protect, self.filter_duration,
            self.filter_manual, self.filter_lp, self.show_raw_overlay,
            self.neighbor_sigmas, self.neighbor_passes,
            self.hampel_window, self.hampel_sigmas,
            self.iqr_window, self.iqr_k, self.iqr_passes,
            self.dur_baseline, self.dur_k, self.dur_min, self.dur_passes,
            self.lp_cutoff,
        ]
        for w in targets:
            w.blockSignals(block)

    def _on_suggest_filter_params(self):
        """Analiza la señal cargada y sugiere parámetros adaptados."""
        if self.current_t is None or self.current_p is None:
            QtWidgets.QMessageBox.information(
                self, "Sugerencias",
                "Primero genera o carga una señal para que el sistema\n"
                "pueda analizarla."
            )
            return

        try:
            sugg = suggest_filter_params(self.current_t, self.current_p)
        except Exception as e:
            QtWidgets.QMessageBox.warning(
                self, "Error en el análisis",
                f"No se pudo analizar la señal:\n{e}"
            )
            return
        if not sugg:
            QtWidgets.QMessageBox.information(
                self, "Sugerencias",
                "La señal es demasiado corta para sugerir parámetros."
            )
            return

        # Aplicar sin disparar callbacks intermedios
        self._block_filter_signals(True)
        try:
            self.neighbor_sigmas.setValue(float(sugg["neighbor_n_sigmas"]))
            self.neighbor_passes.setValue(int(sugg["neighbor_passes"]))
            self.hampel_window.setValue(int(sugg["hampel_window"]))
            self.hampel_sigmas.setValue(float(sugg["hampel_n_sigmas"]))
            self.iqr_window.setValue(int(sugg["iqr_window"]))
            self.iqr_k.setValue(float(sugg["iqr_k"]))
            self.iqr_passes.setValue(int(sugg["iqr_passes"]))
            self.dur_baseline.setValue(int(sugg["dur_baseline"]))
            self.dur_k.setValue(float(sugg["dur_k"]))
            self.dur_min.setValue(float(sugg["dur_min"]))
            self.dur_passes.setValue(int(sugg["dur_passes"]))
            self.lp_cutoff.setValue(float(sugg["lowpass_cutoff"]))
        finally:
            self._block_filter_signals(False)

        # Reaplicar (un solo refresh) y notificar al usuario
        self._on_filter_changed()
        fs = sugg.get("fs", 0)
        density = sugg.get("spike_density", 0) * 100
        sigma = sugg.get("sigma_noise", 0)
        self.mw.status.showMessage(
            f"🪄 Valores sugeridos aplicados  ·  fs={fs:.0f} Hz  ·  "
            f"ruido σ≈{sigma:.4f}  ·  spikes≈{density:.1f}%"
        )

    def _on_reset_filter_params(self):
        """Restaura los parámetros (no las activaciones) a los defaults."""
        defaults = FilterConfig()
        self._block_filter_signals(True)
        try:
            self.neighbor_sigmas.setValue(defaults.neighbor_threshold_sigmas)
            self.neighbor_passes.setValue(defaults.neighbor_max_passes)
            self.hampel_window.setValue(defaults.hampel_window)
            self.hampel_sigmas.setValue(defaults.hampel_n_sigmas)
            self.iqr_window.setValue(defaults.iqr_window)
            self.iqr_k.setValue(defaults.iqr_k)
            self.iqr_passes.setValue(defaults.iqr_max_passes)
            self.iqr_protect.setChecked(defaults.iqr_protect_transients)
            self.dur_baseline.setValue(defaults.duration_baseline_window)
            self.dur_k.setValue(defaults.duration_k_sigmas)
            self.dur_min.setValue(defaults.duration_min_transient_s)
            self.dur_passes.setValue(defaults.duration_max_passes)
            self.lp_cutoff.setValue(defaults.lowpass_cutoff)
        finally:
            self._block_filter_signals(False)
        self._on_filter_changed()
        self.mw.status.showMessage("↺ Parámetros restaurados a sus valores por defecto.")

    def _reapply_filter(self):
        if self.current_p is None or self.current_fs is None:
            return
        self.filter_config = self._build_filter_config_from_ui()
        try:
            self.current_p_filt, self.current_diag = apply_filter_pipeline(
                self.current_p, self.current_fs, self.filter_config,
                t=self.current_t,
            )
        except Exception as e:
            self.current_p_filt = self.current_p.copy()
            self.current_diag = {"error": str(e)}

    def _on_filter_changed(self, *_):
        if self.current_t is None:
            return
        self._reapply_filter()
        self._refresh_views(self._last_label)

    # ---------- callbacks principales ----------
    def on_simulate(self):
        params = TransientParams(
            duration=self.duration.value(), fs=self.fs.value(),
            p0=self.p0.value(), A=self.A.value(), f0=self.f0.value(),
            tau=self.tau.value(), t0=self.t0.value(),
            noise_std=self.noise.value(), bypass=self.bypass.isChecked(),
        )
        t, p = generate_transient(params)
        self.current_t, self.current_p, self.current_fs = t, p, params.fs
        self._reapply_filter()
        label = "Bypass Clandestino" if params.bypass else "Sistema Normal"
        self._refresh_views(label)
        self.mw.status.showMessage("✅  Simulación generada.")

    def on_save_csv(self):
        if self.current_t is None:
            QtWidgets.QMessageBox.warning(self, "Aviso", "No hay señal para guardar.")
            return

        # Decidir qué versión guardar:
        # - Si el master está activo Y hay señal filtrada disponible
        #   distinta de la cruda → guardar la filtrada.
        # - En cualquier otro caso → guardar la cruda.
        filtering_on = self.filter_enable.isChecked()
        has_filtered = self.current_p_filt is not None

        if filtering_on and has_filtered:
            data = self.current_p_filt
            kind_label = "filtrada"
            default_suffix = "_filtrada"
        else:
            data = self.current_p
            kind_label = "cruda"
            default_suffix = ""

        # Sugerir un nombre informativo. Si la señal viene de un CSV,
        # usar su nombre como base.
        default_name = ""
        if hasattr(self, "_last_loaded_path") and self._last_loaded_path:
            base = os.path.splitext(os.path.basename(self._last_loaded_path))[0]
            default_name = f"{base}{default_suffix}.csv"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, f"Guardar CSV ({kind_label})", default_name, SIGNAL_SAVE_FILTER
        )
        if not path:
            return
        try:
            np.savetxt(path, np.column_stack([self.current_t, data]),
                       delimiter=",", header="t,p", comments="")
            self.mw.status.showMessage(
                f"💾  Guardado ({kind_label}): {os.path.basename(path)}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error al guardar", str(e))

    def on_load_csv(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Abrir señal (CSV / TXT / LOG)", "", SIGNAL_FILE_FILTER
        )
        if not path:
            return
        try:
            t, p = load_csv_signal(path)
            self.current_t, self.current_p = t, p
            self.current_fs = infer_fs(t)
            self._last_loaded_path = path  # recordar para el save
            self._reapply_filter()
            self._refresh_views(f"Señal: {os.path.basename(path)}")
            self.mw.status.showMessage(f"📂  Cargado: {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def _refresh_views(self, label: str):
        self._last_label = label
        if self.current_p_filt is None:
            self._reapply_filter()

        t      = self.current_t
        p_raw  = self.current_p
        p      = self.current_p_filt if self.current_p_filt is not None else p_raw
        fs     = self.current_fs
        diag   = self.current_diag or {}

        filtering_on = self.filter_enable.isChecked()
        show_raw     = self.show_raw_overlay.isChecked() and filtering_on

        mask_neighbor = diag.get("neighbor_outliers")
        mask_hampel   = diag.get("hampel_outliers")
        mask_iqr      = diag.get("iqr_outliers")
        mask_dur      = diag.get("duration_outliers")
        mask_man      = diag.get("manual_outliers")
        mask_protect  = diag.get("iqr_protect")
        n_neighbor = int(mask_neighbor.sum()) if mask_neighbor is not None else 0
        n_hampel   = int(mask_hampel.sum())   if mask_hampel   is not None else 0
        n_iqr      = int(mask_iqr.sum())      if mask_iqr      is not None else 0
        n_dur      = int(mask_dur.sum())      if mask_dur      is not None else 0
        n_man      = int(mask_man.sum())      if mask_man      is not None else 0
        n_total    = n_neighbor + n_hampel + n_iqr + n_dur + n_man

        # ── Señal + PSD ──────────────────────────────────────────────
        self.c_signal.clear_axes()
        ax_t, ax_f = self.c_signal.axes[0, 0], self.c_signal.axes[1, 0]

        if show_raw:
            ax_t.plot(t, p_raw, color=COLOR_TEXT_DIM, linewidth=0.7,
                      alpha=0.45, label="Cruda")
        sig_label = "Filtrada" if filtering_on else "Señal"
        ax_t.plot(t, p, color=COLOR_CYAN, linewidth=1.2, label=sig_label)

        # Marcadores diferenciados:
        #   '+' naranja  → vecinos
        #   'x' rojo     → Hampel
        #   '*' magenta  → envolvente IQR (racimo)
        #   'v' amarillo → filtro de duración
        #   'D' cian     → eliminación manual
        if show_raw and mask_neighbor is not None and mask_neighbor.any():
            ax_t.scatter(t[mask_neighbor], p_raw[mask_neighbor],
                         color=COLOR_ORANGE, s=36, marker="+", linewidths=1.6,
                         label=f"Vecinos ({n_neighbor})", zorder=5)
        if show_raw and mask_hampel is not None and mask_hampel.any():
            ax_t.scatter(t[mask_hampel], p_raw[mask_hampel],
                         color=COLOR_DANGER, s=24, marker="x", linewidths=1.4,
                         label=f"Hampel ({n_hampel})", zorder=6)
        if show_raw and mask_iqr is not None and mask_iqr.any():
            ax_t.scatter(t[mask_iqr], p_raw[mask_iqr],
                         color=COLOR_MAGENTA, s=40, marker="*", linewidths=1.0,
                         label=f"IQR ({n_iqr})", zorder=7)
        if show_raw and mask_dur is not None and mask_dur.any():
            ax_t.scatter(t[mask_dur], p_raw[mask_dur],
                         color=COLOR_WARNING, s=28, marker="v", linewidths=0.8,
                         label=f"Duración ({n_dur})", zorder=8)
        if show_raw and mask_man is not None and mask_man.any():
            ax_t.scatter(t[mask_man], p_raw[mask_man],
                         color=COLOR_CYAN, s=46, marker="D",
                         facecolors='none', linewidths=1.2,
                         label=f"Manual ({n_man})", zorder=9)

        # Sombreado opcional de los intervalos manuales activos para que
        # el usuario los vea en el gráfico mientras los edita
        if filtering_on and self.filter_config.manual_enabled:
            for itv in self.filter_config.manual_intervals:
                if not itv.enabled or itv.t_end <= itv.t_start:
                    continue
                ax_t.axvspan(itv.t_start, itv.t_end,
                             color=COLOR_CYAN, alpha=0.08, zorder=1)

        title = label
        # Indicador de frecuencia de muestreo siempre presente cuando hay señal.
        if self.current_fs is not None and self.current_fs > 0:
            title += f"   ·   fs = {self.current_fs} Hz"
        if filtering_on and n_total:
            parts = []
            if n_neighbor: parts.append(f"{n_neighbor} vecinos")
            if n_hampel:   parts.append(f"{n_hampel} Hampel")
            if n_iqr:      parts.append(f"{n_iqr} IQR")
            if n_dur:      parts.append(f"{n_dur} duración")
            if n_man:      parts.append(f"{n_man} manual")
            title += f"   ·   {n_total} pico(s) suprimido(s)  [" + " + ".join(parts) + "]"
        ax_t.set_title(title, fontsize=10, pad=6)
        ax_t.set_xlabel("Tiempo (s)"); ax_t.set_ylabel("Presión (bar)")
        leg = ax_t.legend(loc="upper right", fontsize=8, framealpha=0.6,
                          facecolor=COLOR_PANEL_ALT, edgecolor=COLOR_BORDER)
        if leg:
            for tx in leg.get_texts(): tx.set_color(COLOR_TEXT)

        baseline = np.median(p[: max(1, int(0.05 * len(p)))])
        f_psd, Pxx = sp_signal.welch(p - baseline, fs=fs, nperseg=min(1024, len(p)))
        ax_f.semilogy(f_psd, Pxx, color=COLOR_ORANGE, linewidth=1.2,
                      label=("Filtrada" if filtering_on else "Señal"))
        if show_raw:
            base_r = np.median(p_raw[: max(1, int(0.05 * len(p_raw)))])
            f_r, P_r = sp_signal.welch(p_raw - base_r, fs=fs,
                                       nperseg=min(1024, len(p_raw)))
            ax_f.semilogy(f_r, P_r, color=COLOR_TEXT_DIM, linewidth=0.8,
                          alpha=0.5, label="Cruda")
            leg2 = ax_f.legend(loc="upper right", fontsize=8, framealpha=0.6,
                               facecolor=COLOR_PANEL_ALT, edgecolor=COLOR_BORDER)
            if leg2:
                for tx in leg2.get_texts(): tx.set_color(COLOR_TEXT)
        ax_f.set_title("PSD (Welch)", fontsize=10, pad=6)
        ax_f.set_xlabel("Frecuencia (Hz)"); ax_f.set_ylabel("PSD")
        self.c_signal.draw_idle()

        # ── Espectrograma (sobre señal filtrada) ─────────────────────
        # Usamos reset_figure() en lugar de clear_axes() porque este plot
        # tiene colorbar — si solo limpiásemos el axes principal, los
        # colorbars de redibujos previos se acumularían a la derecha y
        # comprimirían el espectrograma cada vez que cargas una señal.
        self.c_spec.reset_figure()
        ax_s = self.c_spec.axes[0, 0]
        nperseg = min(256, max(64, len(p) // 32))
        f_sg, t_sg, Sxx = sp_signal.spectrogram(p - baseline, fs=fs, nperseg=nperseg)
        im = ax_s.pcolormesh(t_sg, f_sg, 10 * np.log10(Sxx + 1e-12),
                             shading="gouraud", cmap="magma")
        ax_s.set_ylim(0, min(fs / 2, 500))
        spec_title = "Espectrograma (dB)"
        if filtering_on:
            spec_title += " — señal filtrada"
        ax_s.set_title(spec_title, fontsize=10, pad=6)
        ax_s.set_xlabel("Tiempo (s)"); ax_s.set_ylabel("Frecuencia (Hz)")
        cb = self.c_spec.fig.colorbar(im, ax=ax_s, pad=0.01)
        cb.ax.tick_params(colors=COLOR_TEXT_DIM, labelsize=8)
        cb.outline.set_edgecolor(COLOR_BORDER)
        self.c_spec.draw_idle()

        # ── Wavelet (sobre señal filtrada) ───────────────────────────
        self.c_wave.clear_axes()
        ax_w = self.c_wave.axes[0, 0]
        try:
            coeffs = pywt.wavedec(p - baseline, WAVELET_NAME, level=WAVELET_LEVEL)
            offset = 0
            palette = [COLOR_ACCENT, COLOR_CYAN, COLOR_MAGENTA, COLOR_ORANGE, COLOR_SUCCESS]
            for i, c in enumerate(coeffs):
                c = np.asarray(c); c = c / (np.max(np.abs(c)) + 1e-9)
                x = np.linspace(0, t[-1], len(c))
                ax_w.plot(x, c + offset, color=palette[i % len(palette)], linewidth=1.0,
                          label=("cA" if i == 0 else f"cD{WAVELET_LEVEL - i + 1}"))
                offset += 2.3
            ax_w.set_title(f"Wavelet ({WAVELET_NAME}, {WAVELET_LEVEL} niveles)",
                           fontsize=10, pad=6)
            ax_w.set_xlabel("Tiempo (s)"); ax_w.set_yticks([])
            leg = ax_w.legend(loc="upper right", fontsize=8, framealpha=0.6,
                              facecolor=COLOR_PANEL_ALT, edgecolor=COLOR_BORDER)
            for tx in leg.get_texts(): tx.set_color(COLOR_TEXT)
        except Exception as e:
            ax_w.text(0.5, 0.5, f"Wavelet error: {e}", ha="center", va="center",
                      color=COLOR_DANGER, transform=ax_w.transAxes)
        self.c_wave.draw_idle()

        # ── Vista 3D (v4.0, render perezoso) ─────────────────────────
        suffix_3d = "señal filtrada" if filtering_on else ""
        self._3d_payload = (t, p, fs, baseline, suffix_3d)
        self._3d_dirty = True
        # Si el usuario YA está mirando la pestaña 3D, renderizar ahora
        self._maybe_render_3d()

        # ── Cuadro de features (sobre señal filtrada) ────────────────
        feats = extract_features(t, p, fs)
        lines = [f"[{label}]", "-" * 36]
        if filtering_on:
            cfg = self.filter_config
            lines.append("Filtrado: ACTIVO")
            if cfg.neighbor_enabled:
                lines.append(f"  Vecinos n·σ={cfg.neighbor_threshold_sigmas:.1f}"
                             f"  pasadas={cfg.neighbor_max_passes}"
                             f"  → {n_neighbor} pico(s)")
            if cfg.hampel_enabled:
                lines.append(f"  Hampel  w={cfg.hampel_window}"
                             f"  n·σ={cfg.hampel_n_sigmas:.1f}"
                             f"  → {n_hampel} pico(s)")
            if cfg.iqr_enabled:
                prot_pct = 0.0
                if mask_protect is not None and len(mask_protect) > 0:
                    prot_pct = 100.0 * float(np.sum(mask_protect)) / len(mask_protect)
                prot_str = f"  prot={prot_pct:.0f}%" if cfg.iqr_protect_transients else ""
                lines.append(f"  IQR     W={cfg.iqr_window}"
                             f"  k={cfg.iqr_k:.1f}"
                             f"  pas={cfg.iqr_max_passes}"
                             f"{prot_str}"
                             f"  → {n_iqr} pico(s)")
            if cfg.duration_enabled:
                lines.append(f"  Dur.    W={cfg.duration_baseline_window}"
                             f"  k={cfg.duration_k_sigmas:.1f}"
                             f"  min={cfg.duration_min_transient_s*1000:.0f}ms"
                             f"  pas={cfg.duration_max_passes}"
                             f"  → {n_dur} pico(s)")
            if cfg.manual_enabled and cfg.manual_intervals:
                n_active = sum(1 for itv in cfg.manual_intervals if itv.enabled)
                lines.append(f"  Manual  intervalos={n_active}/"
                             f"{len(cfg.manual_intervals)}"
                             f"  → {n_man} pico(s)")
            if cfg.lowpass_enabled:
                lines.append(f"  Butter  fc={cfg.lowpass_cutoff:.0f} Hz"
                             f"  ord={cfg.lowpass_order}")
            lines.append(f"  Total picos suprimidos: {n_total}")
            lines.append("")
        else:
            lines.append("Filtrado: desactivado"); lines.append("")
        for k in sorted(feats.keys()):
            lines.append(f"{k:20s} {feats[k]:>12.4f}")
        self.features_text.setPlainText("\n".join(lines))

    def on_predict(self):
        if self.current_t is None:
            QtWidgets.QMessageBox.warning(self, "Aviso", "Primero genera o carga una señal.")
            return
        if not self.mw.has_any_model():
            QtWidgets.QMessageBox.warning(self, "Aviso",
                                          "No hay modelo cargado ni entrenado.")
            return

        # Predicción se hace sobre la señal filtrada (si filtrado activo)
        if self.current_p_filt is None:
            self._reapply_filter()
        p_for_pred = (self.current_p_filt
                      if self.current_p_filt is not None
                      else self.current_p)

        feats = extract_features(self.current_t, p_for_pred, self.current_fs)
        X, _ = features_to_vector(feats, self.mw.feature_names)
        X = X.reshape(1, -1)
        Xs = self.mw.scaler.transform(X) if self.mw.scaler is not None else X

        # Recolectar predicción y confianza por modelo individual (sin ensemble)
        per_model: Dict[str, Dict[str, Any]] = {}
        for key in self.mw.available_model_keys():
            m = self.mw.models[key]
            try:
                pred = int(m.predict(Xs)[0])
            except Exception:
                continue
            prob = _prob_bypass(m, Xs)
            # Confianza = probabilidad de la clase predicha
            if prob is not None:
                conf = float(prob[0]) if pred == 1 else float(1.0 - prob[0])
            else:
                conf = float("nan")
            per_model[key] = {
                "pred": pred,
                "conf": conf,
                "short": MODEL_SHORT_NAMES[key],
            }

        if not per_model:
            self.badge.set_neutral()
            return

        # Decidir el veredicto final.
        # Regla:
        #   - Si hay 2+ modelos → voting ensemble (soft voting) decide.
        #   - Si hay 1 modelo  → ese modelo decide.
        if len(per_model) >= 2:
            prob_ens = ensemble_prob_bypass(self.mw.models, Xs)
            if prob_ens is not None:
                final_pred = int(prob_ens[0] >= 0.5)
                final_conf = float(prob_ens[0]) if final_pred == 1 else float(1.0 - prob_ens[0])
                source_label = (
                    f"Voting Ensemble  ({'+'.join(per_model[k]['short'] for k in per_model)})"
                )
            else:
                # Fallback raro: ensemble no calculable, usar el primer modelo
                first_key, first_info = next(iter(per_model.items()))
                final_pred = first_info["pred"]
                final_conf = first_info["conf"]
                source_label = MODEL_DISPLAY_NAMES.get(first_key, first_key)
        else:
            # Un solo modelo cargado
            only_key, only_info = next(iter(per_model.items()))
            final_pred = only_info["pred"]
            final_conf = only_info["conf"]
            source_label = MODEL_DISPLAY_NAMES.get(only_key, only_key)

        self.badge.set_decision(
            pred=final_pred, conf=final_conf,
            per_model=per_model,
            source_label=source_label,
        )

        self.mw.status.showMessage("🎯  Predicción completada.")


class SyntheticTrainerTab(QtWidgets.QWidget):
    """Pestaña 2 — entrenar con dataset sintético."""
    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self._build_ui()

    def _build_ui(self):
        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10); root.setSpacing(10)

        # Panel de parámetros (con scroll por la cantidad de controles)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        left = QtWidgets.QWidget(); ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(left)

        grp = QtWidgets.QGroupBox("🧪  Generación de dataset sintético")
        form = QtWidgets.QFormLayout(grp); form.setVerticalSpacing(8)
        self.n_samples = QtWidgets.QSpinBox(); self.n_samples.setRange(100, 20000); self.n_samples.setValue(800)
        self.n_samples.setToolTip("Total de señales a generar (50% bypass / 50% normal).")
        self.duration  = QtWidgets.QDoubleSpinBox(); self.duration.setRange(1.0, 20.0); self.duration.setValue(5.0); self.duration.setSuffix(" s")
        self.duration.setToolTip("Duración de cada señal generada.")
        self.fs        = QtWidgets.QSpinBox(); self.fs.setRange(200, 5000); self.fs.setValue(2000); self.fs.setSuffix(" Hz")
        self.fs.setToolTip("Frecuencia de muestreo de cada señal generada.")
        form.addRow("Nº de muestras:", self.n_samples)
        form.addRow("Duración:", self.duration)
        form.addRow("Muestreo (fs):", self.fs)
        ll.addWidget(grp)

        # Rangos físicos
        self.range_panel = PhysicalRangePanel(
            "🎚️  Rangos físicos del transiente"
        )
        ll.addWidget(self.range_panel)

        # === Modelos a entrenar (RF, SVM, XGB, LGBM) ===
        grp2 = QtWidgets.QGroupBox("🧠  Modelos a entrenar")
        fl = QtWidgets.QFormLayout(grp2); fl.setVerticalSpacing(8)

        self.use_rf   = self._make_model_checkbox("rf",   "Random Forest", default=True)
        self.use_svm  = self._make_model_checkbox("svm",  "SVM (RBF)",     default=True)
        self.use_xgb  = self._make_model_checkbox("xgb",  "XGBoost",       default=False)
        self.use_lgbm = self._make_model_checkbox("lgbm", "LightGBM",      default=False)
        fl.addRow("", self.use_rf)
        fl.addRow("", self.use_svm)
        fl.addRow("", self.use_xgb)
        fl.addRow("", self.use_lgbm)

        self.n_estimators = QtWidgets.QSpinBox(); self.n_estimators.setRange(10, 1000); self.n_estimators.setValue(150)
        self.n_estimators.setToolTip("Aplicado a RF, XGBoost y LightGBM.")
        self.svm_c = QtWidgets.QDoubleSpinBox(); self.svm_c.setRange(0.01, 100.0); self.svm_c.setSingleStep(0.1); self.svm_c.setValue(1.0)
        self.test_size = QtWidgets.QDoubleSpinBox(); self.test_size.setRange(0.05, 0.5); self.test_size.setSingleStep(0.05); self.test_size.setValue(0.25)
        self.lr_boost = QtWidgets.QDoubleSpinBox(); self.lr_boost.setRange(0.001, 1.0); self.lr_boost.setSingleStep(0.01); self.lr_boost.setDecimals(3); self.lr_boost.setValue(0.1)
        self.lr_boost.setToolTip("Tasa de aprendizaje para XGBoost / LightGBM.")
        self.max_depth = QtWidgets.QSpinBox(); self.max_depth.setRange(2, 30); self.max_depth.setValue(6)
        self.max_depth.setToolTip("Profundidad máxima para XGBoost / LightGBM.")

        fl.addRow("Árboles (RF/XGB/LGBM):", self.n_estimators)
        fl.addRow("SVM C:", self.svm_c)
        fl.addRow("Tasa aprendizaje:", self.lr_boost)
        fl.addRow("Max depth:", self.max_depth)
        fl.addRow("Test size:", self.test_size)

        # Calibración
        self.calibrate = QtWidgets.QCheckBox(
            "Calibrar probabilidades (isotonic, recomendado)"
        )
        self.calibrate.setChecked(True)
        self.calibrate.setToolTip(
            "Envuelve cada modelo en CalibratedClassifierCV(method='isotonic').\n"
            "Esto hace que los % de confianza sean realistas (un modelo que\n"
            "muestra 90% acertará realmente cerca del 90% de las veces).\n"
            "Cuesta un poco más de entrenamiento, pero normalmente mejora.\n\n"
            "Si el dataset por clase es < 3 muestras, se omite automáticamente."
        )
        fl.addRow("", self.calibrate)

        ll.addWidget(grp2)

        self.btn_train = QtWidgets.QPushButton("⚡  Entrenar con datos sintéticos")
        self.btn_train.setObjectName("primaryButton"); self.btn_train.setMinimumHeight(40)
        self.btn_train.clicked.connect(self.on_train)
        ll.addWidget(self.btn_train)
        ll.addStretch()

        # Resultados
        right = QtWidgets.QWidget(); rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        grp3 = QtWidgets.QGroupBox("📋  Resumen")
        g3 = QtWidgets.QVBoxLayout(grp3)
        self.summary = QtWidgets.QPlainTextEdit(); self.summary.setReadOnly(True)
        self.summary.setPlainText(
            "Genera un dataset aleatorizando parámetros físicos de cada\n"
            "señal entre los rangos definidos en «Rangos físicos del\n"
            "transiente». Por defecto:\n"
            "  • p₀ ∈ [1.8, 3.5] bar\n"
            "  • A  ∈ [0.3, 1.0]\n"
            "  • f₀ ∈ [10, 70] Hz\n"
            "  • τ  ∈ [0.15, 0.7] s\n"
            "  • t₀ ∈ [0.2, 0.8] s\n"
            "  • σ  ∈ [0.005, 0.02]\n"
            "  • 50 % bypass / 50 % normal\n\n"
            "Tras entrenar, revisa la pestaña «Análisis del modelo»\n"
            "para ver matriz de confusión, importancia y métricas."
        )
        g3.addWidget(self.summary)
        rl.addWidget(grp3)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(scroll); splitter.addWidget(right)
        splitter.setSizes([420, 700])
        root.addWidget(splitter)

    def _make_model_checkbox(self, key: str, label: str,
                              default: bool = False) -> QtWidgets.QCheckBox:
        """Crea un checkbox para activar un modelo. Si la dependencia
        no está instalada (XGB/LGBM), lo deshabilita y explica por qué."""
        cb = QtWidgets.QCheckBox(label)
        if model_is_available(key):
            cb.setChecked(default)
            cb.setEnabled(True)
        else:
            cb.setChecked(False)
            cb.setEnabled(False)
            pkg = "xgboost" if key == "xgb" else "lightgbm"
            cb.setToolTip(
                f"Esta opción requiere la librería '{pkg}'.\n"
                f"Instálala con:  pip install {pkg}"
            )
            cb.setText(f"{label}  (no instalado)")
        return cb

    def _selected_models(self) -> List[str]:
        """Lista de claves de modelos seleccionados en la UI."""
        out = []
        if self.use_rf.isChecked():   out.append("rf")
        if self.use_svm.isChecked():  out.append("svm")
        if self.use_xgb.isChecked():  out.append("xgb")
        if self.use_lgbm.isChecked(): out.append("lgbm")
        return out

    def on_train(self):
        keys = self._selected_models()
        if not keys:
            QtWidgets.QMessageBox.warning(self, "Aviso", "Selecciona al menos un modelo.")
            return
        worker = SyntheticTrainingWorker(
            n_samples=self.n_samples.value(),
            n_estimators=self.n_estimators.value(),
            svm_c=self.svm_c.value(),
            fs=self.fs.value(),
            duration=self.duration.value(),
            test_size=self.test_size.value(),
            models_to_train=keys,
            calibrate=self.calibrate.isChecked(),
            learning_rate=self.lr_boost.value(),
            max_depth=self.max_depth.value(),
            param_ranges=self.range_panel.get_ranges(),
        )
        self.mw.start_training(worker)


class RealTrainerTab(QtWidgets.QWidget):
    """Pestaña 3 — entrenar desde CSVs reales con data augmentation.

    En v3.17 se introduce un QTabWidget interno con dos sub-pestañas:
        1) «Entrenamiento» — UI clásica de entrenamiento (data_no/yes,
           parámetros ML, botón "Aumentar y entrenar").
        2) «Validación Cruzada» — visible solo cuando el usuario activa
           el checkbox «Activar Validación Cruzada». Permite ejecutar
           KFold / StratifiedKFold sobre los mismos datos cargados,
           con análisis avanzado de muestras conflictivas.

    Los datos cargados (data_no, data_yes) se comparten entre las dos
    sub-pestañas — cargar una señal en una pestaña la deja disponible
    para la otra.
    """
    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self.data_no:  List[Tuple[str, np.ndarray, np.ndarray, int]] = []
        self.data_yes: List[Tuple[str, np.ndarray, np.ndarray, int]] = []
        # Estado de validación cruzada
        self._cv_thread: Optional[QtCore.QThread] = None
        self._cv_worker: Optional[CrossValidationWorker] = None
        self.last_cv_result: Optional[CrossValidationResult] = None
        self._build_ui()

    # ------------------------------------------------------------------
    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0); outer.setSpacing(4)

        # Sub-pestañas internas
        self.real_subtabs = QtWidgets.QTabWidget()
        self.real_subtabs.setDocumentMode(True)
        outer.addWidget(self.real_subtabs, 1)

        # Sub-pestaña 1: Entrenamiento clásico
        train_widget = QtWidgets.QWidget()
        self._build_training_ui(train_widget)
        self.real_subtabs.addTab(train_widget, "🚀  Entrenamiento")

        # Sub-pestaña 2: Validación Cruzada — SIEMPRE disponible (v4.2).
        # Antes estaba oculta tras un checkbox «Activar Validación
        # Cruzada»; ahora es una pestaña permanente como cualquier otra.
        self.cv_widget = QtWidgets.QWidget()
        self._build_cv_ui(self.cv_widget)
        self.cv_tab_index = self.real_subtabs.addTab(
            self.cv_widget, "🔬  Validación Cruzada"
        )
        # Poblar el panel «Datos disponibles» con el estado inicial
        self._refresh_cv_data_status()

    # ============================================================
    # SUB-PESTAÑA: VALIDACIÓN CRUZADA
    # ============================================================
    def _build_cv_ui(self, container: QtWidgets.QWidget):
        root = QtWidgets.QVBoxLayout(container)
        root.setContentsMargins(10, 10, 10, 10); root.setSpacing(8)

        # ── Panel superior: configuración ─────────────────────────
        cfg_grp = QtWidgets.QGroupBox("⚙  Configuración de la validación")
        cfg_h = QtWidgets.QHBoxLayout(cfg_grp)
        cfg_h.setSpacing(20)

        # Columna izquierda: estrategia + folds + shuffle/seed
        col_left = QtWidgets.QFormLayout()
        col_left.setVerticalSpacing(6)
        self.cv_strategy = QtWidgets.QComboBox()
        self.cv_strategy.addItems([
            "StratifiedKFold (recomendado)",
            "KFold (simple)",
            "Leave-One-Out (impacto por señal)",
        ])
        self.cv_strategy.setToolTip(
            "StratifiedKFold mantiene la proporción de clases en cada fold —\n"
            "obligatorio si el dataset está desbalanceado.\n\n"
            "KFold hace un split aleatorio simple sin estratificar.\n\n"
            "Leave-One-Out (LOO): para cada señal original se entrena un\n"
            "modelo SIN ella y se compara la accuracy frente al baseline.\n"
            "Identifica señales cuya presencia degrada la generalización\n"
            "(probable etiqueta errónea u outlier real)."
        )
        # Cuando el usuario cambia a LOO, deshabilitar campo n_splits
        # (LOO no usa K).
        self.cv_strategy.currentIndexChanged.connect(self._on_cv_strategy_changed)
        col_left.addRow("Estrategia:", self.cv_strategy)

        self.cv_n_splits = QtWidgets.QSpinBox()
        self.cv_n_splits.setRange(2, 20); self.cv_n_splits.setValue(5)
        self.cv_n_splits.setToolTip(
            "Número de folds. 5 ó 10 son los valores típicos en la literatura.\n"
            "Más folds = estimación más estable pero más tiempo de cómputo."
        )
        col_left.addRow("Nº de folds (K):", self.cv_n_splits)

        self.cv_shuffle = QtWidgets.QCheckBox("Barajar antes de dividir")
        self.cv_shuffle.setChecked(True)
        self.cv_shuffle.setToolTip("Recomendado activar para evitar sesgos por orden de carga.")
        col_left.addRow("", self.cv_shuffle)

        self.cv_seed = QtWidgets.QSpinBox()
        self.cv_seed.setRange(0, 99999); self.cv_seed.setValue(42)
        col_left.addRow("Semilla:", self.cv_seed)

        cfg_h.addLayout(col_left, 1)

        # Columna central: modelos
        col_mid = QtWidgets.QVBoxLayout(); col_mid.setSpacing(4)
        col_mid.addWidget(QtWidgets.QLabel("<b>Modelos a validar:</b>"))
        self.cv_use_rf   = self._make_model_checkbox("rf",   "Random Forest", default=True)
        self.cv_use_svm  = self._make_model_checkbox("svm",  "SVM (RBF)",     default=True)
        self.cv_use_xgb  = self._make_model_checkbox("xgb",  "XGBoost",       default=False)
        self.cv_use_lgbm = self._make_model_checkbox("lgbm", "LightGBM",      default=False)
        col_mid.addWidget(self.cv_use_rf)
        col_mid.addWidget(self.cv_use_svm)
        col_mid.addWidget(self.cv_use_xgb)
        col_mid.addWidget(self.cv_use_lgbm)
        col_mid.addStretch(1)
        cfg_h.addLayout(col_mid, 1)

        # Columna derecha: hint con datos disponibles + botón
        col_right = QtWidgets.QVBoxLayout(); col_right.setSpacing(6)
        self.cv_data_status = QtWidgets.QLabel("...")
        self.cv_data_status.setWordWrap(True)
        self.cv_data_status.setStyleSheet(
            f"color:{COLOR_TEXT_DIM}; font-size:9pt;"
            f"background-color:{COLOR_PANEL_ALT}; "
            f"border:1px solid {COLOR_BORDER}; border-radius:4px; padding:6px;"
        )
        col_right.addWidget(self.cv_data_status)

        self.cv_use_existing_params = QtWidgets.QCheckBox(
            "Usar hiperparámetros del entrenamiento"
        )
        self.cv_use_existing_params.setChecked(True)
        self.cv_use_existing_params.setToolTip(
            "Si está activado, la CV usa los mismos n_estimators, SVM C,\n"
            "learning rate, max_depth, calibración, n_aug y target_total\n"
            "configurados en la sub-pestaña «Entrenamiento».\n"
            "Si lo desactivas podrás configurar valores propios para la CV."
        )
        col_right.addWidget(self.cv_use_existing_params)

        col_right.addStretch(1)
        cfg_h.addLayout(col_right, 1)

        root.addWidget(cfg_grp)

        # Botones de acción
        btn_row = QtWidgets.QHBoxLayout(); btn_row.setSpacing(8)
        self.btn_cv_run = QtWidgets.QPushButton("🚀  Ejecutar Validación Cruzada")
        self.btn_cv_run.setObjectName("primaryButton")
        self.btn_cv_run.setMinimumHeight(40)
        self.btn_cv_run.clicked.connect(self._on_cv_run)
        btn_row.addWidget(self.btn_cv_run, 1)

        self.btn_cv_cancel = QtWidgets.QPushButton("⏹  Cancelar")
        self.btn_cv_cancel.setEnabled(False)
        self.btn_cv_cancel.clicked.connect(self._on_cv_cancel)
        btn_row.addWidget(self.btn_cv_cancel)

        self.btn_cv_export = QtWidgets.QPushButton("⬇  Exportar resultados (CSV)")
        self.btn_cv_export.setEnabled(False)
        self.btn_cv_export.clicked.connect(self._on_cv_export)
        btn_row.addWidget(self.btn_cv_export)
        root.addLayout(btn_row)

        # Barra de progreso + label
        progress_row = QtWidgets.QHBoxLayout(); progress_row.setSpacing(6)
        self.cv_progress = QtWidgets.QProgressBar()
        self.cv_progress.setRange(0, 100); self.cv_progress.setValue(0)
        self.cv_progress_label = QtWidgets.QLabel("Listo.")
        self.cv_progress_label.setStyleSheet(f"color:{COLOR_TEXT_DIM};")
        progress_row.addWidget(self.cv_progress, 1)
        progress_row.addWidget(self.cv_progress_label)
        root.addLayout(progress_row)

        # ── Panel inferior: resultados (sub-tabs internas) ────────
        self.cv_results_tabs = QtWidgets.QTabWidget()
        self.cv_results_tabs.setDocumentMode(True)
        root.addWidget(self.cv_results_tabs, 1)

        # Sub-tab: Resumen (incluye Ranking de modelos al final)
        self.cv_results_tabs.addTab(self._build_cv_summary_tab(),
                                     "📊  Resumen por modelo")
        # Sub-tab: Heatmap de estabilidad
        self.cv_results_tabs.addTab(self._build_cv_heatmap_tab(),
                                     "🌡️  Heatmap de estabilidad")
        # Sub-tab: Ranking conflictivos
        self.cv_results_tabs.addTab(self._build_cv_conflict_tab(),
                                     "⚠️  Muestras conflictivas")
        # Sub-tab: Outliers
        self.cv_results_tabs.addTab(self._build_cv_outliers_tab(),
                                     "🚫  Outliers detectados")
        # Sub-tab: Impacto LOO (solo útil cuando estrategia = "loo")
        self.cv_results_tabs.addTab(self._build_cv_loo_tab(),
                                     "🔄  Impacto LOO")
        # Sub-tab: Log
        self.cv_results_tabs.addTab(self._build_cv_log_tab(),
                                     "📝  Log")

    def _build_cv_summary_tab(self) -> QtWidgets.QWidget:
        """
        Resumen por modelo + métricas detalladas por fold + sección de
        ranking automático con vista tipo acordeón.

        Esta es la pestaña central de la sub-pestaña de Validación
        Cruzada. Tiene tres secciones apiladas verticalmente dentro de
        un QScrollArea para que el usuario pueda hacer scroll si hay
        muchos folds × algoritmos:
            1) Tabla resumen por modelo (1 fila por algoritmo).
            2) Tabla con las métricas detalladas por fold.
            3) Ranking automático: tree-view tipo acordeón donde cada
               nodo top-level es el "mejor modelo del algoritmo"
               (entrenado en todo el dataset) y los hijos colapsados
               son los K modelos individuales de cada fold.
        """
        outer = QtWidgets.QWidget()
        outer_lay = QtWidgets.QVBoxLayout(outer)
        outer_lay.setContentsMargins(0, 0, 0, 0); outer_lay.setSpacing(0)

        # Hacemos toda la página scrollable porque con 4 modelos × K folds
        # la sección de ranking puede crecer mucho.
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        outer_lay.addWidget(scroll, 1)

        page = QtWidgets.QWidget()
        scroll.setWidget(page)
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(10)

        # ── Sección 1: tabla resumen por modelo ───────────────────
        s1_title = QtWidgets.QLabel(
            f"<b style='color:{COLOR_ACCENT}; font-size:11pt;'>"
            f"📊 Resumen agregado por modelo</b>"
        )
        lay.addWidget(s1_title)
        self.cv_summary_table = QtWidgets.QTableWidget(0, 0)
        self.cv_summary_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
        )
        self.cv_summary_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )
        self.cv_summary_table.setAlternatingRowColors(True)
        self.cv_summary_table.verticalHeader().setVisible(False)
        self.cv_summary_table.setMinimumHeight(120)
        lay.addWidget(self.cv_summary_table)

        # ── Sección 2: tabla detallada por fold ───────────────────
        s2_title = QtWidgets.QLabel(
            f"<b style='color:{COLOR_ACCENT}; font-size:11pt;'>"
            f"📋 Métricas detalladas por fold</b>"
        )
        lay.addWidget(s2_title)
        self.cv_folds_table = QtWidgets.QTableWidget(0, 0)
        self.cv_folds_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
        )
        self.cv_folds_table.setAlternatingRowColors(True)
        self.cv_folds_table.verticalHeader().setVisible(False)
        self.cv_folds_table.setMinimumHeight(180)
        lay.addWidget(self.cv_folds_table)

        # ── Sección 3: Ranking automático (con acordeón) ──────────
        s3_title = QtWidgets.QLabel(
            f"<b style='color:{COLOR_ACCENT}; font-size:11pt;'>"
            f"🏆 Ranking automático de modelos</b>"
        )
        lay.addWidget(s3_title)
        s3_help = QtWidgets.QLabel(
            "Por defecto se muestra el <b>mejor modelo de cada algoritmo</b> "
            "(entrenado sobre todo el dataset). Expande cada fila ▶ para ver "
            "los modelos individuales entrenados en cada fold. "
            "Cualquier modelo puede seleccionarse para descargarlo o "
            "cargarlo al programa."
        )
        s3_help.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:9pt;")
        s3_help.setWordWrap(True)
        lay.addWidget(s3_help)

        # Criterio de ranking
        crit_row = QtWidgets.QHBoxLayout(); crit_row.setSpacing(6)
        crit_row.addWidget(QtWidgets.QLabel("<b>Ordenar por:</b>"))
        self.cv_ranking_criterion = QtWidgets.QComboBox()
        self.cv_ranking_criterion.addItems([
            "Accuracy (media)",
            "Estabilidad (1 − std de accuracy)",
            "Robustez (acc del peor fold)",
            "Velocidad de inferencia (1 / tiempo)",
            "F1-score (media)",
            "ROC-AUC (media)",
            "Score compuesto (recomendado)",
        ])
        self.cv_ranking_criterion.setToolTip(
            "Criterio que ordena los algoritmos top-level (top a bottom).\n"
            "Los modelos por fold dentro de cada acordeón siempre se\n"
            "ordenan por accuracy del fold (mejor primero).\n\n"
            "Accuracy media : promedio de aciertos en todos los folds.\n"
            "Estabilidad    : penaliza modelos con alta varianza entre folds.\n"
            "Robustez       : prioriza el modelo con el peor-caso menos malo.\n"
            "Inferencia     : el más rápido en entrenar (proxy de predicción).\n"
            "F1, ROC-AUC    : alternativas a accuracy.\n"
            "Score compuesto: 0.50·acc + 0.20·(1−std) + 0.20·F1 + 0.10·AUC."
        )
        self.cv_ranking_criterion.currentIndexChanged.connect(
            self._refresh_cv_ranking
        )
        crit_row.addWidget(self.cv_ranking_criterion, 1)
        lay.addLayout(crit_row)

        # Tree-widget tipo acordeón
        self.cv_ranking_tree = QtWidgets.QTreeWidget()
        self.cv_ranking_tree.setColumnCount(0)   # se ajusta dinámicamente
        self.cv_ranking_tree.setAlternatingRowColors(True)
        self.cv_ranking_tree.setUniformRowHeights(True)
        self.cv_ranking_tree.setRootIsDecorated(True)
        self.cv_ranking_tree.setExpandsOnDoubleClick(True)
        self.cv_ranking_tree.setMinimumHeight(220)
        # Multi-selección: Ctrl+click añade/quita, Shift+click selecciona rango.
        # Permite cargar varios algoritmos a la vez (ej. RF + XGB).
        self.cv_ranking_tree.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection
        )
        self.cv_ranking_tree.itemSelectionChanged.connect(
            self._on_cv_ranking_selection
        )
        lay.addWidget(self.cv_ranking_tree)

        # Etiqueta + botones de selección
        sel_row = QtWidgets.QHBoxLayout(); sel_row.setSpacing(8)
        self.cv_ranking_selected_lbl = QtWidgets.QLabel(
            "Selecciona un modelo del ranking para descargarlo o cargarlo."
        )
        self.cv_ranking_selected_lbl.setStyleSheet(
            f"color:{COLOR_TEXT_DIM}; font-style:italic;"
        )
        self.cv_ranking_selected_lbl.setWordWrap(True)
        sel_row.addWidget(self.cv_ranking_selected_lbl, 1)
        lay.addLayout(sel_row)

        btn_row = QtWidgets.QHBoxLayout(); btn_row.setSpacing(8)
        self.btn_cv_download = QtWidgets.QPushButton(
            "⬇  Descargar selección"
        )
        self.btn_cv_download.setEnabled(False)
        self.btn_cv_download.setToolTip(
            "Guarda el modelo (o modelos) seleccionado(s) como un .joblib.\n"
            "Si seleccionas varios (Ctrl+click), se guarda un único .joblib\n"
            "con todos los modelos seleccionados — listos para ensemble."
        )
        self.btn_cv_download.clicked.connect(self._on_cv_ranking_download)
        btn_row.addWidget(self.btn_cv_download)

        self.btn_cv_apply = QtWidgets.QPushButton("📥  Cargar selección al programa")
        self.btn_cv_apply.setObjectName("primaryButton")
        self.btn_cv_apply.setEnabled(False)
        self.btn_cv_apply.setToolTip(
            "Carga el modelo (o modelos) seleccionado(s) como el modelo activo.\n"
            "Selección múltiple: Ctrl+click para añadir, Shift+click para rango.\n"
            "Ej: seleccionando SVM + XGB se cargan ambos y el voting ensemble\n"
            "los usa automáticamente para las predicciones."
        )
        self.btn_cv_apply.clicked.connect(self._on_cv_ranking_apply)
        btn_row.addWidget(self.btn_cv_apply)

        # Botón "Descargar mejor modelo" con popup multi-selección
        self.btn_cv_download_best = QtWidgets.QPushButton(
            "⭐  Descargar mejor modelo…"
        )
        self.btn_cv_download_best.setEnabled(False)
        self.btn_cv_download_best.setToolTip(
            "Abre un diálogo con checkboxes para elegir uno o varios\n"
            "algoritmos. Si seleccionas varios, se descarga un único\n"
            ".joblib que contiene los mejores modelos seleccionados\n"
            "(p.ej. rf+xgboost) — listos para ensemble en el programa."
        )
        self.btn_cv_download_best.clicked.connect(self._on_cv_download_best)
        btn_row.addWidget(self.btn_cv_download_best)
        lay.addLayout(btn_row)

        # Estado interno
        # Cuando el usuario selecciona en el tree, guardamos:
        #   - tipo: "full" (mejor del algoritmo) o "fold" (un fold concreto)
        #   - key:  rf/svm/xgb/lgbm
        #   - fold_idx: número de fold (o None si es full)
        self._cv_ranking_sel: Optional[Dict[str, Any]] = None
        self._cv_full_models: Dict[str, Any] = {}
        self._cv_full_scaler: Optional[StandardScaler] = None

        lay.addStretch(0)
        return outer

    def _build_cv_heatmap_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(4)
        info = QtWidgets.QLabel(
            "Heatmap de aciertos/errores por muestra. Verde = predicción "
            "correcta, rojo = error. Las muestras conflictivas aparecen "
            "como rayas rojas verticales (todos los modelos fallan)."
        )
        info.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:9pt;")
        info.setWordWrap(True)
        lay.addWidget(info)
        self.cv_heatmap_canvas = PlotCanvas(nrows=1, ncols=1)
        lay.addWidget(NavigationToolbar(self.cv_heatmap_canvas, self))
        lay.addWidget(self.cv_heatmap_canvas, 1)
        self.cv_heatmap_canvas.show_empty(
            "Ejecuta una validación cruzada para ver el heatmap."
        )
        return page

    def _build_cv_conflict_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(6)

        info = QtWidgets.QLabel(
            "Ranking de muestras donde los modelos más fallaron durante la CV.\n"
            "Cada muestra aparece en exactamente UN fold como validación. "
            "El «Índice de confiabilidad» = fracción de modelos que la "
            "clasificaron correctamente en ese fold."
        )
        info.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:9pt;")
        info.setWordWrap(True)
        lay.addWidget(info)

        # Filtro de cantidad
        filt = QtWidgets.QHBoxLayout(); filt.setSpacing(6)
        filt.addWidget(QtWidgets.QLabel("Mostrar top:"))
        self.cv_conflict_top = QtWidgets.QSpinBox()
        self.cv_conflict_top.setRange(5, 5000); self.cv_conflict_top.setValue(50)
        self.cv_conflict_top.valueChanged.connect(self._refresh_cv_conflict)
        filt.addWidget(self.cv_conflict_top)
        filt.addStretch(1)
        lay.addLayout(filt)

        self.cv_conflict_table = QtWidgets.QTableWidget(0, 0)
        self.cv_conflict_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
        )
        self.cv_conflict_table.setSortingEnabled(True)
        self.cv_conflict_table.setAlternatingRowColors(True)
        self.cv_conflict_table.verticalHeader().setVisible(False)
        lay.addWidget(self.cv_conflict_table, 1)
        return page

    def _build_cv_outliers_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(6)

        info = QtWidgets.QLabel(
            "<b>Detección automática de outliers / muestras anómalas.</b><br>"
            "Una muestra es marcada como outlier cuando la mayoría de modelos "
            "la clasificaron incorrectamente durante la validación cruzada. "
            "Esto puede indicar: (a) etiqueta errónea, (b) señal ruidosa o "
            "corrupta, (c) caso fuera de la distribución del dataset.<br><br>"
            "Acción recomendada: revisar manualmente estas señales y, si "
            "tienen problemas, eliminarlas o re-etiquetarlas antes de reentrenar."
        )
        info.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:9pt;")
        info.setWordWrap(True)
        lay.addWidget(info)

        # Umbral de outlier
        thr = QtWidgets.QHBoxLayout()
        thr.addWidget(QtWidgets.QLabel("Umbral mínimo de error (% de modelos que fallaron):"))
        self.cv_outlier_threshold = QtWidgets.QSpinBox()
        self.cv_outlier_threshold.setRange(1, 100); self.cv_outlier_threshold.setValue(50)
        self.cv_outlier_threshold.setSuffix(" %")
        self.cv_outlier_threshold.setToolTip(
            "Umbral mínimo de tasa de error para considerar una muestra como\n"
            "candidata a outlier. Valores típicos: 50% (mayoría de modelos\n"
            "fallaron), 25% (algunos modelos fallaron), 1% (cualquier error)."
        )
        self.cv_outlier_threshold.valueChanged.connect(self._refresh_cv_outliers)
        thr.addWidget(self.cv_outlier_threshold)
        thr.addStretch(1)
        lay.addLayout(thr)

        self.cv_outlier_summary = QtWidgets.QLabel("Sin datos.")
        self.cv_outlier_summary.setStyleSheet(f"color:{COLOR_TEXT}; font-weight:600;")
        self.cv_outlier_summary.setWordWrap(True)
        lay.addWidget(self.cv_outlier_summary)

        self.cv_outlier_table = QtWidgets.QTableWidget(0, 0)
        self.cv_outlier_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
        )
        self.cv_outlier_table.setSortingEnabled(True)
        self.cv_outlier_table.setAlternatingRowColors(True)
        self.cv_outlier_table.verticalHeader().setVisible(False)
        lay.addWidget(self.cv_outlier_table, 1)

        # Botón para eliminar del dataset las señales marcadas como outlier.
        # Trabaja sobre archivos originales (no augmentaciones): un único
        # CSV puede haber generado varias filas augmentadas; al borrarlo
        # se eliminan todas sus muestras de golpe.
        rm_row = QtWidgets.QHBoxLayout(); rm_row.setSpacing(8)
        rm_row.addStretch(1)
        self.btn_remove_outliers = QtWidgets.QPushButton(
            "🗑️  Eliminar outliers del dataset"
        )
        self.btn_remove_outliers.setEnabled(False)
        self.btn_remove_outliers.setToolTip(
            "Elimina de data_no y data_yes las señales ORIGINALES "
            "(archivos cargados) cuyas augmentaciones aparecen como "
            "outliers según el umbral configurado.\n\n"
            "Útil para limpiar etiquetas erróneas o señales corruptas\n"
            "antes de reentrenar."
        )
        self.btn_remove_outliers.clicked.connect(self._on_remove_outliers)
        rm_row.addWidget(self.btn_remove_outliers)
        lay.addLayout(rm_row)
        return page

    def _build_cv_loo_tab(self) -> QtWidgets.QWidget:
        """Sub-tab para mostrar el impacto LOO por señal."""
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(6)

        info = QtWidgets.QLabel(
            "<b>Análisis Leave-One-Out por señal original.</b><br>"
            "Para cada señal del dataset se entrena un modelo sin ella y se "
            "compara la accuracy del modelo «sin esa señal» frente al "
            "baseline (modelo con todas las señales).<br><br>"
            "<b>Interpretación de Δ (delta = acc_sin − acc_con):</b><br>"
            f"&nbsp;&nbsp;• <span style='color:{COLOR_DANGER};'>Δ &gt; 0</span> "
            "→ quitar la señal MEJORA el modelo (tóxica/sospechosa).<br>"
            f"&nbsp;&nbsp;• <span style='color:{COLOR_TEXT_DIM};'>Δ ≈ 0</span> "
            "→ señal neutra (su presencia no afecta).<br>"
            f"&nbsp;&nbsp;• <span style='color:{COLOR_SUCCESS};'>Δ &lt; 0</span> "
            "→ la señal aporta información útil al modelo.<br><br>"
            "Esta estrategia solo se ejecuta cuando seleccionas «Leave-One-Out» "
            "en el combo de estrategia y pulsas «Ejecutar Validación Cruzada»."
        )
        info.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:9pt;")
        info.setWordWrap(True)
        lay.addWidget(info)

        # Resumen
        self.cv_loo_summary = QtWidgets.QLabel(
            "Aún no se ha ejecutado un análisis LOO."
        )
        self.cv_loo_summary.setStyleSheet(
            f"color:{COLOR_TEXT}; font-weight:600; "
            f"background-color:{COLOR_PANEL_ALT}; padding:8px; "
            f"border:1px solid {COLOR_BORDER}; border-radius:4px;"
        )
        self.cv_loo_summary.setWordWrap(True)
        lay.addWidget(self.cv_loo_summary)

        # Tabla
        self.cv_loo_table = QtWidgets.QTableWidget(0, 0)
        self.cv_loo_table.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
        )
        self.cv_loo_table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )
        self.cv_loo_table.setSortingEnabled(True)
        self.cv_loo_table.setAlternatingRowColors(True)
        self.cv_loo_table.verticalHeader().setVisible(False)
        lay.addWidget(self.cv_loo_table, 1)
        return page

    def _build_cv_log_tab(self) -> QtWidgets.QWidget:
        page = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(page)
        lay.setContentsMargins(8, 8, 8, 8); lay.setSpacing(4)
        self.cv_log_view = QtWidgets.QPlainTextEdit()
        self.cv_log_view.setReadOnly(True)
        self.cv_log_view.setStyleSheet(
            f"background-color:{COLOR_PANEL_ALT}; "
            f"color:{COLOR_TEXT}; font-family:Consolas, 'Courier New', monospace;"
        )
        self.cv_log_view.setPlainText(
            "Log de validación cruzada — los detalles de cada fold "
            "aparecerán aquí.\n"
        )
        lay.addWidget(self.cv_log_view, 1)
        return page

    def _on_cv_strategy_changed(self, idx: int):
        """Deshabilita controles irrelevantes según la estrategia elegida."""
        txt = self.cv_strategy.currentText()
        is_loo = txt.startswith("Leave-One-Out")
        # En LOO no hay K folds (cada señal es un fold)
        if hasattr(self, "cv_n_splits"):
            self.cv_n_splits.setEnabled(not is_loo)
            self.cv_n_splits.setToolTip(
                "No aplica a Leave-One-Out (cada señal se trata como un fold)."
                if is_loo else
                "Número de folds. 5 ó 10 son los valores típicos en la literatura.\n"
                "Más folds = estimación más estable pero más tiempo de cómputo."
            )

    def _refresh_cv_data_status(self):
        """Refresca el cuadro de 'Datos disponibles' en la cabecera CV."""
        n_no, n_yes = len(self.data_no), len(self.data_yes)
        if n_no == 0 and n_yes == 0:
            self.cv_data_status.setText(
                "<b>Datos disponibles:</b><br>"
                "<span style='color:" + COLOR_DANGER + ";'>Sin señales cargadas.</span><br>"
                "Carga señales en la pestaña «Entrenamiento» primero."
            )
        else:
            self.cv_data_status.setText(
                "<b>Datos disponibles:</b><br>"
                f"• {n_no} señales Normal<br>"
                f"• {n_yes} señales Bypass<br>"
                "Tras augmentación se generará un dataset balanceado."
            )

    # ------------------------------------------------------------------
    # Ejecución de la CV
    # ------------------------------------------------------------------
    def _on_cv_run(self):
        if not self.data_no or not self.data_yes:
            QtWidgets.QMessageBox.warning(
                self, "Datos insuficientes",
                "Necesitas al menos una señal en cada clase (Normal y Bypass).\n"
                "Carga los archivos en la sub-pestaña «Entrenamiento»."
            )
            return

        keys = []
        for cb, k in [(self.cv_use_rf, "rf"), (self.cv_use_svm, "svm"),
                       (self.cv_use_xgb, "xgb"), (self.cv_use_lgbm, "lgbm")]:
            if cb.isChecked() and model_is_available(k):
                keys.append(k)
        if not keys:
            QtWidgets.QMessageBox.warning(
                self, "Sin modelos", "Selecciona al menos un modelo para validar."
            )
            return

        # Estrategia: usar lo que el usuario ya seleccionó en el combo de
        # la cabecera. No hace falta volver a preguntar: la selección
        # está visible y se puede cambiar antes de pulsar el botón.
        strategy_text = self.cv_strategy.currentText()
        if strategy_text.startswith("KFold"):
            strategy = "kfold"
        elif strategy_text.startswith("Leave-One-Out"):
            strategy = "loo"
        else:
            strategy = "stratified"

        n = self.cv_n_splits.value()
        # Validación rápida: cada clase debe tener al menos n señales
        # originales para que stratified pueda dividir cleanmente.
        if strategy == "stratified" and (len(self.data_no) < n or len(self.data_yes) < n):
            ret = QtWidgets.QMessageBox.warning(
                self, "Aviso de splits",
                f"Tienes {len(self.data_no)} Normal y {len(self.data_yes)} Bypass.\n"
                f"Con {n} folds y StratifiedKFold puede no haber suficientes "
                f"muestras por clase en cada fold. ¿Continuar de todos modos?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            )
            if ret != QtWidgets.QMessageBox.Yes:
                return

        # Construir config: leer hiperparámetros de la pestaña entrenamiento
        # si el usuario lo pidió, o usar defaults
        if self.cv_use_existing_params.isChecked():
            cfg = CrossValidationConfig(
                strategy=strategy,
                n_splits=n,
                shuffle=self.cv_shuffle.isChecked(),
                random_state=self.cv_seed.value(),
                models_to_validate=keys,
                n_estimators=self.n_estimators.value(),
                svm_c=self.svm_c.value(),
                learning_rate=self.lr_boost.value(),
                max_depth=self.max_depth.value(),
                calibrate=self.calibrate.isChecked(),
                n_aug=self.n_aug.value(),
                target_total=self.target_total.value(),
            )
        else:
            # Defaults conservadores
            cfg = CrossValidationConfig(
                strategy=strategy,
                n_splits=n,
                shuffle=self.cv_shuffle.isChecked(),
                random_state=self.cv_seed.value(),
                models_to_validate=keys,
            )

        # Arrancar el worker en su thread
        self._cv_thread = QtCore.QThread(self)
        self._cv_worker = CrossValidationWorker(
            self.data_no, self.data_yes, cfg
        )
        self._cv_worker.moveToThread(self._cv_thread)
        self._cv_thread.started.connect(self._cv_worker.run)
        self._cv_worker.progress.connect(self._on_cv_progress)
        self._cv_worker.finished.connect(self._on_cv_finished)
        self._cv_worker.failed.connect(self._on_cv_failed)
        self._cv_worker.log.connect(self._cv_append_log)
        self._cv_worker.finished.connect(self._cv_thread.quit)
        self._cv_worker.failed.connect(self._cv_thread.quit)
        self._cv_thread.finished.connect(self._cv_worker.deleteLater)
        self._cv_thread.finished.connect(self._cv_thread.deleteLater)
        self._cv_thread.finished.connect(self._on_cv_thread_finished)

        # UI: lock controls
        self.btn_cv_run.setEnabled(False)
        self.btn_cv_cancel.setEnabled(True)
        self.btn_cv_export.setEnabled(False)
        self.cv_progress.setValue(0)
        self.cv_progress_label.setText("Iniciando…")
        self.cv_log_view.clear()
        self.cv_log_view.appendPlainText(
            f"════════════════════════════════════════════════════\n"
            f"  Validación Cruzada — {datetime.datetime.now().isoformat(timespec='seconds')}\n"
            f"════════════════════════════════════════════════════"
        )
        self._cv_thread.start()

    @QtCore.pyqtSlot(int, str)
    def _on_cv_progress(self, pct: int, msg: str):
        FX.animate_progress(self.cv_progress, pct)   # v4.0: suavizado
        self.cv_progress_label.setText(msg)

    @QtCore.pyqtSlot(str)
    def _cv_append_log(self, msg: str):
        self.cv_log_view.appendPlainText(msg)

    @QtCore.pyqtSlot(object)
    def _on_cv_finished(self, result: CrossValidationResult):
        self.last_cv_result = result
        self.btn_cv_run.setEnabled(True)
        self.btn_cv_cancel.setEnabled(False)
        self.btn_cv_export.setEnabled(True)
        self.cv_progress.setValue(100)
        self.cv_progress_label.setText("✅  Completada.")

        # Mensaje del status bar: distinto para LOO porque ahí no hay "K folds"
        if result.config.strategy == "loo":
            n_tox = sum(1 for r in result.loo_impact if r.delta > 0.005)
            self.mw.status.showMessage(
                f"🔬  Leave-One-Out finalizado · "
                f"{len(result.loo_impact)} señales analizadas · "
                f"{n_tox} sospechosa(s) · {result.total_time_s:.1f}s."
            )
        else:
            self.mw.status.showMessage(
                f"🔬  Validación cruzada finalizada — "
                f"{len(result.per_model)} modelos × {result.config.n_splits} folds "
                f"en {result.total_time_s:.1f}s."
            )

        # Entrenar UN modelo final por cada clave en todo el dataset
        # augmentado (sin hold-out). Estos son los modelos que se ofrecen
        # para descarga/aplicación desde la pestaña Ranking. Se entrenan
        # AHORA con los mismos hiperparámetros que los folds.
        try:
            self._train_full_models_for_ranking(result)
        except Exception as e:
            self._cv_append_log(
                f"⚠  No se pudieron entrenar los modelos finales para ranking: {e}"
            )

        self._render_cv_results(result)

    def _train_full_models_for_ranking(self, result: CrossValidationResult):
        """
        Re-entrena cada modelo seleccionado sobre TODOS los datos
        augmentados (sin hold-out). Esto sirve para que el botón
        «Descargar modelo» y «Cargar al programa» puedan devolver un
        modelo realmente usable, no uno de un fold cualquiera.

        Reusa la misma augmentación que el worker, pero rehacemos el
        cálculo de features aquí en el hilo principal porque queremos
        que `_cv_full_models` contenga objetos sklearn-entrenados antes
        de que el usuario pueda interactuar con la pestaña Ranking.
        """
        cfg = result.config
        # Para evitar re-extraer features de cero, podemos pedirle al
        # worker que los devuelva. Pero el worker ya terminó. Hacemos
        # la extracción de nuevo (es relativamente barata).
        feats_no = []
        for path, t, p, fs in self.data_no:
            for t2, p2 in augment_single_signal(t, p, n_aug=cfg.n_aug):
                try:
                    feats_no.append(extract_features(t2, p2, fs))
                except Exception:
                    pass
        feats_yes = []
        for path, t, p, fs in self.data_yes:
            for t2, p2 in augment_single_signal(t, p, n_aug=cfg.n_aug):
                try:
                    feats_yes.append(extract_features(t2, p2, fs))
                except Exception:
                    pass
        if not feats_no or not feats_yes:
            self._cv_full_models = {}
            self._cv_full_scaler = None
            return

        feature_names = sorted(feats_no[0].keys())
        X_no  = np.array([[f[k] for k in feature_names] for f in feats_no])
        X_yes = np.array([[f[k] for k in feature_names] for f in feats_yes])
        half = cfg.target_total // 2
        rng = np.random.default_rng(cfg.random_state)

        def resample(Xg, N):
            idx = rng.choice(len(Xg), size=N, replace=(len(Xg) < N))
            return Xg[idx]
        Xb_no  = resample(X_no, half)
        Xb_yes = resample(X_yes, cfg.target_total - half)
        X = np.vstack([Xb_no, Xb_yes])
        y = np.hstack([
            np.zeros(len(Xb_no), dtype=int),
            np.ones(len(Xb_yes), dtype=int),
        ])
        perm = rng.permutation(len(y))
        X, y = X[perm], y[perm]

        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        self._cv_full_scaler = scaler
        self._cv_feature_names = feature_names
        self._cv_full_models = {}
        for key in result.per_model.keys():
            try:
                base = _make_classifier(
                    key,
                    n_estimators=cfg.n_estimators,
                    svm_c=cfg.svm_c,
                    learning_rate=cfg.learning_rate,
                    max_depth=cfg.max_depth,
                )
                clf = _calibrated_wrap(
                    base, Xs, y, enable=cfg.calibrate, cv=3
                )
                clf.fit(Xs, y)
                self._cv_full_models[key] = clf
            except Exception as e:
                self._cv_append_log(
                    f"⚠  Modelo final {key} falló: {e}"
                )

    @QtCore.pyqtSlot(str)
    def _on_cv_failed(self, tb: str):
        self.btn_cv_run.setEnabled(True)
        self.btn_cv_cancel.setEnabled(False)
        self.cv_progress.setValue(0)
        self.cv_progress_label.setText("❌  Error.")
        self._cv_append_log("\n❌  ERROR EN LA VALIDACIÓN:\n" + tb)
        QtWidgets.QMessageBox.critical(self, "Error en validación cruzada", tb)

    @QtCore.pyqtSlot()
    def _on_cv_thread_finished(self):
        self._cv_thread = None
        self._cv_worker = None

    def _on_cv_cancel(self):
        if self._cv_worker is not None:
            self._cv_worker.cancel()
            self.cv_progress_label.setText("Cancelando…")
            self._cv_append_log("⏹  Cancelación solicitada por el usuario.")

    def _on_cv_export(self):
        if self.last_cv_result is None:
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exportar resultados de CV", "cv_results.csv",
            "CSV (*.csv);;Todos (*)"
        )
        if not path:
            return
        try:
            self._export_cv_csv(self.last_cv_result, path)
            self.mw.status.showMessage(
                f"💾  Resultados de CV exportados: {os.path.basename(path)}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error al exportar", str(e))

    # ------------------------------------------------------------------
    # Render de resultados
    # ------------------------------------------------------------------
    def _render_cv_results(self, result: CrossValidationResult):
        is_loo = (result.config.strategy == "loo")
        self._render_cv_summary(result)
        self._render_cv_folds(result)

        # En LOO no hay heatmap, ni muestras conflictivas, ni outliers
        # por fold (no hay "predicciones por muestra"). Marcamos esos
        # paneles como N/A.
        if not is_loo:
            self._render_cv_heatmap(result)
            self._refresh_cv_conflict()
            self._refresh_cv_outliers()
        else:
            self.cv_heatmap_canvas.show_empty(
                "El heatmap no aplica para Leave-One-Out\n"
                "(no hay validación por fold)."
            )
            # Limpiar tablas
            self.cv_conflict_table.setRowCount(0)
            self.cv_outlier_table.setRowCount(0)
            self.lbl_per_sample_summary_check()
            self.cv_outlier_summary.setText(
                "El análisis de outliers por fold no aplica para LOO. "
                "Consulta «🔄 Impacto LOO» para identificar señales sospechosas."
            )

        # LOO siempre se refresca (vacío si no aplica)
        self._refresh_cv_loo(result)
        # El ranking se refresca a partir del criterio actual
        self._refresh_cv_ranking()

        # Saltar a la sub-tab más relevante según el modo
        if is_loo:
            # buscar el índice del tab "Impacto LOO"
            for i in range(self.cv_results_tabs.count()):
                if "Impacto" in self.cv_results_tabs.tabText(i):
                    self.cv_results_tabs.setCurrentIndex(i)
                    return
        # Modo KFold/Stratified → al resumen
        self.cv_results_tabs.setCurrentIndex(0)

    def lbl_per_sample_summary_check(self):
        """Helper trivial — placeholder por si en el futuro hay que
        limpiar más estado. Por ahora no hace nada."""
        return

    def _render_cv_summary(self, result: CrossValidationResult):
        """Tabla principal con métricas agregadas por modelo."""
        headers = [
            "Modelo", "Folds",
            "Accuracy", "Std Acc",
            "Precision", "Recall", "F1", "ROC-AUC",
            "Mejor fold (acc)", "Peor fold (acc)",
            "Tiempo total (s)",
        ]
        models = list(result.per_model.values())
        self.cv_summary_table.setSortingEnabled(False)
        self.cv_summary_table.setColumnCount(len(headers))
        self.cv_summary_table.setHorizontalHeaderLabels(headers)
        self.cv_summary_table.setRowCount(len(models))

        ph = self.cv_summary_table.horizontalHeader()
        for c in range(len(headers)):
            ph.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        ph.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)

        def make_item(text: str, value=None, color=None, bold=False):
            if value is not None:
                it = _NumericTableItem(text, float(value))
            else:
                it = QtWidgets.QTableWidgetItem(text)
            it.setTextAlignment(QtCore.Qt.AlignCenter)
            if color is not None:
                it.setForeground(QtGui.QBrush(QtGui.QColor(color)))
            if bold:
                f = it.font(); f.setBold(True); it.setFont(f)
            return it

        for i, mr in enumerate(models):
            display = MODEL_DISPLAY_NAMES.get(mr.model_key, mr.model_key)
            self.cv_summary_table.setItem(i, 0, make_item(display, bold=True, color=COLOR_ACCENT))
            self.cv_summary_table.setItem(i, 1, make_item(str(len(mr.folds)), value=len(mr.folds)))
            self.cv_summary_table.setItem(i, 2, make_item(
                f"{mr.mean_accuracy:.4f}", value=mr.mean_accuracy
            ))
            self.cv_summary_table.setItem(i, 3, make_item(
                f"{mr.std_accuracy:.4f}", value=mr.std_accuracy
            ))
            self.cv_summary_table.setItem(i, 4, make_item(
                f"{mr.mean_precision:.4f}", value=mr.mean_precision
            ))
            self.cv_summary_table.setItem(i, 5, make_item(
                f"{mr.mean_recall:.4f}", value=mr.mean_recall
            ))
            self.cv_summary_table.setItem(i, 6, make_item(
                f"{mr.mean_f1:.4f}", value=mr.mean_f1
            ))
            auc = mr.mean_roc_auc
            self.cv_summary_table.setItem(i, 7, make_item(
                f"{auc:.4f}" if auc is not None else "—",
                value=(auc if auc is not None else 0)
            ))
            best = mr.best_fold
            worst = mr.worst_fold
            self.cv_summary_table.setItem(i, 8, make_item(
                f"#{best.fold_index} ({best.accuracy:.4f})" if best else "—",
                value=(best.accuracy if best else 0)
            ))
            self.cv_summary_table.setItem(i, 9, make_item(
                f"#{worst.fold_index} ({worst.accuracy:.4f})" if worst else "—",
                value=(worst.accuracy if worst else 0)
            ))
            self.cv_summary_table.setItem(i, 10, make_item(
                f"{mr.total_time_s:.2f}", value=mr.total_time_s
            ))
        self.cv_summary_table.setSortingEnabled(True)

    def _render_cv_folds(self, result: CrossValidationResult):
        """Tabla con una fila por modelo×fold."""
        headers = ["Modelo", "Fold", "Acc", "Prec", "Recall", "F1",
                   "ROC-AUC", "n_train", "n_val", "Tiempo (s)"]
        # Total de filas
        total = sum(len(mr.folds) for mr in result.per_model.values())
        self.cv_folds_table.setSortingEnabled(False)
        self.cv_folds_table.setColumnCount(len(headers))
        self.cv_folds_table.setHorizontalHeaderLabels(headers)
        self.cv_folds_table.setRowCount(total)

        ph = self.cv_folds_table.horizontalHeader()
        for c in range(len(headers)):
            ph.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)

        def num_item(value, fmt="{:.4f}", color=None):
            it = _NumericTableItem(fmt.format(value), float(value))
            it.setTextAlignment(QtCore.Qt.AlignCenter)
            if color is not None:
                it.setForeground(QtGui.QBrush(QtGui.QColor(color)))
            return it

        row = 0
        for mr in result.per_model.values():
            short = MODEL_SHORT_NAMES.get(mr.model_key, mr.model_key)
            for f in mr.folds:
                it_m = QtWidgets.QTableWidgetItem(short)
                it_m.setTextAlignment(QtCore.Qt.AlignCenter)
                it_m.setForeground(QtGui.QBrush(QtGui.QColor(COLOR_ACCENT)))
                self.cv_folds_table.setItem(row, 0, it_m)
                self.cv_folds_table.setItem(row, 1, num_item(f.fold_index, "{:.0f}"))
                self.cv_folds_table.setItem(row, 2, num_item(f.accuracy))
                self.cv_folds_table.setItem(row, 3, num_item(f.precision))
                self.cv_folds_table.setItem(row, 4, num_item(f.recall))
                self.cv_folds_table.setItem(row, 5, num_item(f.f1))
                if f.roc_auc is not None:
                    self.cv_folds_table.setItem(row, 6, num_item(f.roc_auc))
                else:
                    it = QtWidgets.QTableWidgetItem("—")
                    it.setTextAlignment(QtCore.Qt.AlignCenter)
                    it.setForeground(QtGui.QBrush(QtGui.QColor(COLOR_TEXT_DIM)))
                    self.cv_folds_table.setItem(row, 6, it)
                self.cv_folds_table.setItem(row, 7, num_item(f.n_train, "{:.0f}"))
                self.cv_folds_table.setItem(row, 8, num_item(f.n_val, "{:.0f}"))
                self.cv_folds_table.setItem(row, 9, num_item(f.train_time_s, "{:.2f}"))
                row += 1
        self.cv_folds_table.setSortingEnabled(True)

    def _render_cv_heatmap(self, result: CrossValidationResult):
        """
        Heatmap (modelos × muestras) coloreado por acierto/error.
        Verde = acierto, rojo = error. Las muestras conflictivas
        aparecen como columnas rojas verticales.
        """
        self.cv_heatmap_canvas.reset_figure()
        ax = self.cv_heatmap_canvas.axes[0, 0]

        models = list(result.per_model.values())
        if not models or result.y_true is None:
            self.cv_heatmap_canvas.show_empty("Sin datos.")
            return

        y_true = result.y_true
        # Matriz: filas=modelos, columnas=muestras, valor=1 si correcto, 0 si error
        n_samples = len(y_true)
        n_models = len(models)
        # Ordenar muestras por tasa de error promedio para que las
        # conflictivas se agrupen visualmente a la derecha.
        err_per_sample = np.zeros(n_samples, dtype=float)
        for mr in models:
            if mr.val_preds is not None:
                err_per_sample += (mr.val_preds != y_true).astype(float)
        err_per_sample /= max(1, n_models)
        order = np.argsort(err_per_sample)  # ascendente: aciertos a la izq, errores a la der

        matrix = np.zeros((n_models, n_samples), dtype=float)
        for i, mr in enumerate(models):
            if mr.val_preds is None:
                continue
            correct = (mr.val_preds == y_true).astype(int)
            matrix[i] = correct[order]

        # Colormap: rojo (0=error) → verde (1=correcto)
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list(
            "rg", [COLOR_DANGER, "#3a3a3a", COLOR_SUCCESS]
        )
        im = ax.imshow(matrix, aspect="auto", interpolation="nearest",
                        cmap=cmap, vmin=0, vmax=1)
        ax.set_yticks(range(n_models))
        ax.set_yticklabels(
            [MODEL_DISPLAY_NAMES.get(mr.model_key, mr.model_key) for mr in models]
        )
        ax.set_xlabel("Muestras (ordenadas: aciertos ← → errores)")
        ax.set_title(
            f"Heatmap de aciertos por modelo  ·  "
            f"{n_samples} muestras  ·  ordenadas por dificultad",
            fontsize=10, pad=8,
        )
        # Hide x-ticks (too many samples)
        ax.set_xticks([])
        self.cv_heatmap_canvas.fig.tight_layout()
        self.cv_heatmap_canvas.draw_idle()

    def _compute_sample_stats(self, result: CrossValidationResult):
        """
        Devuelve para cada muestra:
            - n_models_correct, n_models_wrong, reliability (correctos/total)
            - mean_prob_bypass, std_prob_bypass entre modelos
            - lista de modelos que la clasificaron mal
        """
        y_true = result.y_true
        if y_true is None:
            return []
        n = len(y_true)
        models = list(result.per_model.values())
        n_models = len(models)
        out = []
        for i in range(n):
            wrong_models = []
            probs = []
            for mr in models:
                if mr.val_preds is None: continue
                if mr.val_preds[i] != y_true[i]:
                    wrong_models.append(MODEL_SHORT_NAMES.get(mr.model_key, mr.model_key))
                if mr.val_probs is not None and not np.isnan(mr.val_probs[i]):
                    probs.append(float(mr.val_probs[i]))
            n_correct = n_models - len(wrong_models)
            reliability = n_correct / max(1, n_models)
            out.append({
                "index": i,
                "name": (result.sample_origin[i]
                          if i < len(result.sample_origin) else f"#{i}"),
                "true": int(y_true[i]),
                "n_correct": n_correct,
                "n_wrong": len(wrong_models),
                "wrong_list": wrong_models,
                "reliability": reliability,
                "mean_prob_bypass": float(np.mean(probs)) if probs else None,
                "std_prob_bypass":  float(np.std(probs))  if len(probs) >= 2 else None,
            })
        return out

    def _refresh_cv_conflict(self):
        if self.last_cv_result is None:
            return
        stats = self._compute_sample_stats(self.last_cv_result)
        # Ordenar por reliability ascendente (peores primero)
        stats.sort(key=lambda s: (s["reliability"], s["index"]))
        top = self.cv_conflict_top.value()
        rows = stats[:top]

        headers = ["#", "Señal", "Real", "Aciertos",
                   "Errores", "Confiabilidad",
                   "Prob bypass (media)", "Prob bypass (std)",
                   "Modelos que fallaron"]
        labels = {0: "Normal", 1: "Bypass"}

        self.cv_conflict_table.setSortingEnabled(False)
        self.cv_conflict_table.setColumnCount(len(headers))
        self.cv_conflict_table.setHorizontalHeaderLabels(headers)
        self.cv_conflict_table.setRowCount(len(rows))
        ph = self.cv_conflict_table.horizontalHeader()
        for c in range(len(headers)):
            ph.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        ph.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        ph.setSectionResizeMode(8, QtWidgets.QHeaderView.Stretch)

        for i, s in enumerate(rows):
            def cell(text, value=None, color=None, bold=False):
                if value is not None:
                    it = _NumericTableItem(text, float(value))
                else:
                    it = QtWidgets.QTableWidgetItem(text)
                it.setTextAlignment(QtCore.Qt.AlignCenter)
                if color is not None:
                    it.setForeground(QtGui.QBrush(QtGui.QColor(color)))
                if bold:
                    f = it.font(); f.setBold(True); it.setFont(f)
                return it

            rel = s["reliability"]
            rel_color = COLOR_DANGER if rel < 0.5 else (COLOR_WARNING if rel < 0.8 else COLOR_SUCCESS)
            self.cv_conflict_table.setItem(i, 0, cell(str(i+1), value=i+1))
            it_name = QtWidgets.QTableWidgetItem(s["name"])
            it_name.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            self.cv_conflict_table.setItem(i, 1, it_name)
            self.cv_conflict_table.setItem(i, 2, cell(
                labels[s["true"]],
                color=(COLOR_DANGER if s["true"] == 1 else COLOR_SUCCESS),
                bold=True,
            ))
            self.cv_conflict_table.setItem(i, 3, cell(
                str(s["n_correct"]), value=s["n_correct"], color=COLOR_SUCCESS
            ))
            self.cv_conflict_table.setItem(i, 4, cell(
                str(s["n_wrong"]), value=s["n_wrong"],
                color=(COLOR_DANGER if s["n_wrong"] > 0 else COLOR_TEXT_DIM),
                bold=(s["n_wrong"] > 0),
            ))
            self.cv_conflict_table.setItem(i, 5, cell(
                f"{rel*100:.1f}%", value=rel*100, color=rel_color, bold=True,
            ))
            if s["mean_prob_bypass"] is not None:
                self.cv_conflict_table.setItem(i, 6, cell(
                    f"{s['mean_prob_bypass']:.3f}",
                    value=s["mean_prob_bypass"],
                ))
            else:
                self.cv_conflict_table.setItem(i, 6, cell("—", color=COLOR_TEXT_DIM))
            if s["std_prob_bypass"] is not None:
                self.cv_conflict_table.setItem(i, 7, cell(
                    f"{s['std_prob_bypass']:.3f}",
                    value=s["std_prob_bypass"],
                ))
            else:
                self.cv_conflict_table.setItem(i, 7, cell("—", color=COLOR_TEXT_DIM))
            if s["wrong_list"]:
                wlist = ", ".join(s["wrong_list"])
                wlist_color = COLOR_DANGER
            else:
                wlist = "— (sin errores)"
                wlist_color = COLOR_SUCCESS
            self.cv_conflict_table.setItem(i, 8, cell(
                wlist, color=wlist_color,
            ))
        self.cv_conflict_table.setSortingEnabled(True)

    def _refresh_cv_outliers(self):
        if self.last_cv_result is None:
            return
        stats = self._compute_sample_stats(self.last_cv_result)
        # Umbral: tasa de error >= threshold/100
        thr = self.cv_outlier_threshold.value() / 100.0
        outliers = [s for s in stats if (1.0 - s["reliability"]) >= thr]
        outliers.sort(key=lambda s: s["reliability"])

        n_total = len(stats)
        n_out = len(outliers)
        if n_out == 0:
            self.cv_outlier_summary.setText(
                f"✅ Ningún outlier detectado con umbral ≥ {thr*100:.0f}%. "
                f"Total analizado: {n_total} muestras."
            )
        else:
            pct = (n_out / max(1, n_total)) * 100
            self.cv_outlier_summary.setText(
                f"⚠️  {n_out} de {n_total} muestras ({pct:.1f}%) son candidatas a outlier "
                f"(≥ {thr*100:.0f}% de los modelos las clasificó mal). "
                f"Revisar manualmente."
            )

        headers = ["Señal", "Clase real", "Tasa de error",
                   "Modelos que fallaron"]
        labels = {0: "Normal", 1: "Bypass"}
        self.cv_outlier_table.setSortingEnabled(False)
        self.cv_outlier_table.setColumnCount(len(headers))
        self.cv_outlier_table.setHorizontalHeaderLabels(headers)
        self.cv_outlier_table.setRowCount(len(outliers))
        ph = self.cv_outlier_table.horizontalHeader()
        ph.setSectionResizeMode(0, QtWidgets.QHeaderView.Stretch)
        for c in (1, 2):
            ph.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        ph.setSectionResizeMode(3, QtWidgets.QHeaderView.Stretch)

        for i, s in enumerate(outliers):
            err_rate = 1.0 - s["reliability"]
            it_name = QtWidgets.QTableWidgetItem(s["name"])
            it_name.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            self.cv_outlier_table.setItem(i, 0, it_name)

            it_real = QtWidgets.QTableWidgetItem(labels[s["true"]])
            it_real.setTextAlignment(QtCore.Qt.AlignCenter)
            it_real.setForeground(QtGui.QBrush(QtGui.QColor(
                COLOR_DANGER if s["true"] == 1 else COLOR_SUCCESS
            )))
            f = it_real.font(); f.setBold(True); it_real.setFont(f)
            self.cv_outlier_table.setItem(i, 1, it_real)

            it_err = _NumericTableItem(f"{err_rate*100:.1f}%", err_rate*100)
            it_err.setTextAlignment(QtCore.Qt.AlignCenter)
            it_err.setForeground(QtGui.QBrush(QtGui.QColor(COLOR_DANGER)))
            fo = it_err.font(); fo.setBold(True); it_err.setFont(fo)
            self.cv_outlier_table.setItem(i, 2, it_err)

            wlist = ", ".join(s["wrong_list"])
            self.cv_outlier_table.setItem(i, 3, QtWidgets.QTableWidgetItem(wlist))
        self.cv_outlier_table.setSortingEnabled(True)

        # Habilitar el botón de eliminar solo cuando hay outliers detectados
        # Y los archivos originales correspondientes aún están en data_no/data_yes
        # (el cómputo concreto se hace dentro de _on_remove_outliers).
        self.btn_remove_outliers.setEnabled(n_out > 0)
        if n_out > 0:
            self.btn_remove_outliers.setText(
                f"🗑️  Eliminar outliers del dataset  ({n_out} candidata{'s' if n_out != 1 else ''})"
            )
        else:
            self.btn_remove_outliers.setText(
                "🗑️  Eliminar outliers del dataset"
            )

    def _on_remove_outliers(self):
        """
        Elimina del dataset los archivos originales cuyas muestras
        (originales o augmentadas) aparecen como outliers según el
        umbral actual.

        Nota importante: el análisis CV trabaja sobre muestras augmentadas
        («señal_001.csv» y «señal_001.csv [aug 3]»). Aquí mapeamos de
        vuelta al ARCHIVO ORIGINAL (todo lo que está antes del primer
        «[aug»). Si CUALQUIERA de las augmentaciones de un archivo
        original es outlier, removemos el archivo entero — es la decisión
        más segura: si una augmentación es problemática, probablemente la
        señal madre también lo sea, y dejar las otras augmentaciones
        sueltas no aporta.
        """
        if self.last_cv_result is None:
            return
        stats = self._compute_sample_stats(self.last_cv_result)
        thr = self.cv_outlier_threshold.value() / 100.0
        outliers = [s for s in stats if (1.0 - s["reliability"]) >= thr]
        if not outliers:
            QtWidgets.QMessageBox.information(
                self, "Eliminar outliers",
                "No hay outliers detectados con el umbral actual."
            )
            return

        # Extraer nombre base del archivo (quitar el sufijo " [aug N]")
        def base_filename(name: str) -> str:
            idx = name.find(" [aug")
            return name if idx < 0 else name[:idx]

        offending_names: set = set(base_filename(o["name"]) for o in outliers)

        # Buscar esos nombres en data_no y data_yes (los archivos cargados
        # guardan la ruta completa; comparamos contra el basename)
        removed_no_paths: List[str] = []
        removed_yes_paths: List[str] = []

        def filter_class(group: list) -> Tuple[list, List[str]]:
            keep, removed = [], []
            for entry in group:
                path = entry[0]
                if os.path.basename(path) in offending_names:
                    removed.append(path)
                else:
                    keep.append(entry)
            return keep, removed

        new_no, removed_no_paths = filter_class(self.data_no)
        new_yes, removed_yes_paths = filter_class(self.data_yes)
        total_to_remove = len(removed_no_paths) + len(removed_yes_paths)

        if total_to_remove == 0:
            QtWidgets.QMessageBox.warning(
                self, "Eliminar outliers",
                "Los outliers detectados ya NO están en el dataset cargado.\n\n"
                "Esto pasa cuando: (a) ya los eliminaste antes y los nombres "
                "de la tabla son obsoletos, (b) el dataset fue regenerado o "
                "reordenado entre la CV y ahora."
            )
            return

        # Confirmación con lista previa
        preview_no = "\n".join(
            f"  • {os.path.basename(p)}" for p in removed_no_paths[:10]
        )
        preview_yes = "\n".join(
            f"  • {os.path.basename(p)}" for p in removed_yes_paths[:10]
        )
        if len(removed_no_paths) > 10:
            preview_no += f"\n  … y {len(removed_no_paths) - 10} más"
        if len(removed_yes_paths) > 10:
            preview_yes += f"\n  … y {len(removed_yes_paths) - 10} más"

        msg = (
            f"Se eliminarán <b>{total_to_remove}</b> archivo(s) del dataset:<br><br>"
            f"<b>Clase Normal:</b> {len(removed_no_paths)} archivo(s)"
            + (f"<pre style='color:#aaa;'>{preview_no}</pre>" if preview_no else "<br>")
            + f"<b>Clase Bypass:</b> {len(removed_yes_paths)} archivo(s)"
            + (f"<pre style='color:#aaa;'>{preview_yes}</pre>" if preview_yes else "<br>")
            + "<br>Después de eliminar deberías volver a entrenar / validar.<br><br>"
            + "¿Continuar?"
        )
        ret = QtWidgets.QMessageBox.question(
            self, "Confirmar eliminación", msg,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if ret != QtWidgets.QMessageBox.Yes:
            return

        # Aplicar
        self.data_no  = new_no
        self.data_yes = new_yes

        # Refrescar la UI: listas de archivos, contadores, panel CV
        self.list_no.clear()
        for path, _, _, _ in self.data_no:
            self.list_no.addItem(os.path.basename(path))
        self.list_yes.clear()
        for path, _, _, _ in self.data_yes:
            self.list_yes.addItem(os.path.basename(path))
        self._update_counts()

        # Limpiar la tabla de outliers (los nombres ya no son válidos
        # porque el CV anterior se hizo sobre un dataset que ya no existe).
        self.cv_outlier_table.setRowCount(0)
        self.cv_outlier_summary.setText(
            f"🗑️  {total_to_remove} archivo(s) eliminado(s). "
            f"Vuelve a ejecutar la Validación Cruzada para reanalizar "
            f"el dataset reducido."
        )
        self.btn_remove_outliers.setEnabled(False)
        self.btn_remove_outliers.setText(
            "🗑️  Eliminar outliers del dataset"
        )

        self.mw.status.showMessage(
            f"🗑️  Eliminados {total_to_remove} archivo(s) marcados como outlier."
        )

    # ------------------------------------------------------------------
    # SUB-TAB: IMPACTO LEAVE-ONE-OUT
    # ------------------------------------------------------------------
    def _refresh_cv_loo(self, result: Optional[CrossValidationResult] = None):
        """
        Llena la tabla de Impacto LOO con un registro por señal original.

        Cuando el último análisis NO fue LOO, deja la tabla vacía y un
        mensaje recordatorio en el resumen.
        """
        if result is None:
            result = self.last_cv_result
        if result is None:
            return

        if not result.loo_impact:
            self.cv_loo_summary.setText(
                "El análisis Leave-One-Out no se ejecutó en la última corrida. "
                "Selecciona «Leave-One-Out» en el combo de estrategia y vuelve "
                "a pulsar «Ejecutar Validación Cruzada» para rellenar esta tabla."
            )
            self.cv_loo_table.setRowCount(0)
            self.cv_loo_table.setColumnCount(0)
            return

        # Resumen
        records = list(result.loo_impact)
        baseline = result.loo_baseline_acc or 0.0
        # Tóxicas: delta significativo (>0.005 para evitar ruido numérico)
        toxic = [r for r in records if r.delta > 0.005]
        beneficial = [r for r in records if r.delta < -0.005]
        neutral = len(records) - len(toxic) - len(beneficial)

        self.cv_loo_summary.setText(
            f"<b>Baseline (todas las señales):</b> acc = {baseline:.4f}<br>"
            f"<b>Señales analizadas:</b> {len(records)}<br>"
            f"&nbsp;&nbsp;• <span style='color:{COLOR_DANGER};'>{len(toxic)} tóxicas</span> "
            f"(quitarlas mejora el modelo)<br>"
            f"&nbsp;&nbsp;• <span style='color:{COLOR_TEXT_DIM};'>{neutral} neutras</span><br>"
            f"&nbsp;&nbsp;• <span style='color:{COLOR_SUCCESS};'>{len(beneficial)} útiles</span> "
            f"(aportan información)"
        )

        # Ordenar de mayor delta a menor (tóxicas primero)
        records.sort(key=lambda r: -r.delta)

        headers = ["Rank", "Señal", "Clase",
                   "Acc con", "Acc sin", "Δ (sin − con)", "Interpretación"]
        labels = {0: "Normal", 1: "Bypass"}

        self.cv_loo_table.setSortingEnabled(False)
        self.cv_loo_table.setColumnCount(len(headers))
        self.cv_loo_table.setHorizontalHeaderLabels(headers)
        self.cv_loo_table.setRowCount(len(records))
        ph = self.cv_loo_table.horizontalHeader()
        for c in range(len(headers)):
            ph.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)
        ph.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        ph.setSectionResizeMode(6, QtWidgets.QHeaderView.Stretch)

        def cell(text, value=None, color=None, bold=False,
                  align=QtCore.Qt.AlignCenter):
            if value is not None:
                it = _NumericTableItem(text, float(value))
            else:
                it = QtWidgets.QTableWidgetItem(text)
            it.setTextAlignment(align)
            if color is not None:
                it.setForeground(QtGui.QBrush(QtGui.QColor(color)))
            if bold:
                f = it.font(); f.setBold(True); it.setFont(f)
            return it

        for i, r in enumerate(records):
            if r.delta > 0.005:
                interp = "⚠  Quitarla mejora el modelo (sospechosa)"
                dcolor = COLOR_DANGER
            elif r.delta < -0.005:
                interp = "✓  Aporta información útil"
                dcolor = COLOR_SUCCESS
            else:
                interp = "—  Neutra"
                dcolor = COLOR_TEXT_DIM

            self.cv_loo_table.setItem(i, 0, cell(str(i+1), value=i+1))

            it_name = QtWidgets.QTableWidgetItem(r.signal_name)
            it_name.setTextAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter)
            self.cv_loo_table.setItem(i, 1, it_name)

            self.cv_loo_table.setItem(i, 2, cell(
                labels[r.class_label],
                color=(COLOR_DANGER if r.class_label == 1 else COLOR_SUCCESS),
                bold=True,
            ))
            self.cv_loo_table.setItem(i, 3, cell(
                f"{r.acc_with:.4f}", value=r.acc_with
            ))
            self.cv_loo_table.setItem(i, 4, cell(
                f"{r.acc_without:.4f}", value=r.acc_without
            ))
            sign = "+" if r.delta > 0 else ""
            self.cv_loo_table.setItem(i, 5, cell(
                f"{sign}{r.delta:.4f}", value=r.delta,
                color=dcolor, bold=True,
            ))
            self.cv_loo_table.setItem(i, 6, cell(
                interp, color=dcolor,
                align=QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            ))
        self.cv_loo_table.setSortingEnabled(True)

    # ------------------------------------------------------------------
    # SUB-TAB: AUTOMATIC MODEL RANKING
    # ------------------------------------------------------------------
    def _compute_ranking_scores(self, result: CrossValidationResult,
                                  criterion: str) -> List[Dict[str, Any]]:
        """
        Calcula el score por modelo para el criterio dado y devuelve
        una lista ordenada (mejor primero) de dicts con todas las
        métricas relevantes para mostrar en la tabla.
        """
        rows = []
        for key, mr in result.per_model.items():
            acc      = mr.mean_accuracy
            std      = mr.std_accuracy
            f1       = mr.mean_f1
            auc      = mr.mean_roc_auc or 0.0
            worst    = mr.worst_fold.accuracy if mr.worst_fold else acc
            total_t  = mr.total_time_s
            speed    = (1.0 / total_t) if total_t > 0 else 0.0
            # Score compuesto: balance entre acc, estabilidad, F1 y AUC
            composite = 0.50*acc + 0.20*(1.0 - std) + 0.20*f1 + 0.10*auc

            # Criterio elegido por el usuario
            if criterion.startswith("Accuracy"):
                score = acc
            elif criterion.startswith("Estabilidad"):
                score = 1.0 - std
            elif criterion.startswith("Robustez"):
                score = worst
            elif criterion.startswith("Velocidad"):
                score = speed
            elif criterion.startswith("F1"):
                score = f1
            elif criterion.startswith("ROC-AUC"):
                score = auc
            else:
                score = composite

            rows.append({
                "key": key,
                "display": MODEL_DISPLAY_NAMES.get(key, key),
                "score": score,
                "acc": acc,
                "std": std,
                "f1": f1,
                "auc": auc,
                "worst": worst,
                "time": total_t,
                "composite": composite,
            })
        rows.sort(key=lambda r: -r["score"])
        return rows

    def _refresh_cv_ranking(self):
        """
        Llena el QTreeWidget del ranking según el criterio actual.

        Estructura del tree:
            ► Random Forest [BEST]   acc=...  std=...  F1=...  AUC=...
                ├ Fold 1            acc=...
                ├ Fold 2            acc=...
                └ ...
            ► XGBoost      [BEST]
                ├ ...
            ► ...

        Los nodos top-level representan el "mejor modelo del algoritmo"
        (entrenado en TODO el dataset, sin hold-out). Sus hijos son los
        K modelos individuales de cada fold.
        """
        if self.last_cv_result is None:
            self.cv_ranking_tree.clear()
            self.cv_ranking_tree.setColumnCount(0)
            return

        criterion = self.cv_ranking_criterion.currentText()
        algo_rows = self._compute_ranking_scores(self.last_cv_result, criterion)

        headers = ["Modelo", "Score / Acc", "Std", "F1", "AUC",
                   "Peor fold", "Tiempo (s)", "Tipo"]
        self.cv_ranking_tree.setColumnCount(len(headers))
        self.cv_ranking_tree.setHeaderLabels(headers)
        self.cv_ranking_tree.clear()

        for h_idx in range(len(headers)):
            self.cv_ranking_tree.header().setSectionResizeMode(
                h_idx, QtWidgets.QHeaderView.ResizeToContents
            )
        self.cv_ranking_tree.header().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch
        )

        for rank, r in enumerate(algo_rows):
            key = r["key"]
            mr  = self.last_cv_result.per_model[key]
            display = r["display"]
            best_available = key in self._cv_full_models

            # Item top-level: el "mejor del algoritmo"
            cols = [
                f"#{rank+1}  ⭐ {display}  [BEST]",
                f"{r['score']:.4f}",
                f"{r['std']:.4f}",
                f"{r['f1']:.4f}",
                f"{r['auc']:.4f}" if r['auc'] > 0 else "—",
                f"{r['worst']:.4f}",
                f"{r['time']:.2f}",
                "Full dataset" if best_available else "no entrenado",
            ]
            top = QtWidgets.QTreeWidgetItem(cols)
            # Marcar rank 1 con color destacado
            color_top = COLOR_SUCCESS if rank == 0 else COLOR_ACCENT
            f = top.font(0); f.setBold(True); top.setFont(0, f)
            top.setForeground(0, QtGui.QBrush(QtGui.QColor(color_top)))
            top.setForeground(1, QtGui.QBrush(QtGui.QColor(color_top)))
            # Metadatos para la selección
            top.setData(0, QtCore.Qt.UserRole, {
                "type": "full",
                "key":  key,
                "available": best_available,
            })
            if not best_available:
                # Sin modelo full entrenado → italica + gris
                fi = top.font(0); fi.setItalic(True); top.setFont(0, fi)
                for c in range(len(cols)):
                    top.setForeground(c, QtGui.QBrush(
                        QtGui.QColor(COLOR_TEXT_DIM)
                    ))
            self.cv_ranking_tree.addTopLevelItem(top)

            # Hijos: los K modelos por fold, ordenados por accuracy
            folds_sorted = sorted(mr.folds, key=lambda f: -f.accuracy)
            for f in folds_sorted:
                fold_cols = [
                    f"    Fold {f.fold_index}",
                    f"{f.accuracy:.4f}",
                    "—",   # std no aplica a un solo fold
                    f"{f.f1:.4f}",
                    f"{f.roc_auc:.4f}" if f.roc_auc is not None else "—",
                    f"{f.accuracy:.4f}",   # un fold ES su propio peor caso
                    f"{f.train_time_s:.2f}",
                    "Fold individual",
                ]
                child = QtWidgets.QTreeWidgetItem(fold_cols)
                child.setForeground(0, QtGui.QBrush(QtGui.QColor(COLOR_TEXT_DIM)))
                # Color de la accuracy según rendimiento absoluto
                acc_color = (COLOR_SUCCESS if f.accuracy >= 0.95 else
                              (COLOR_WARNING if f.accuracy >= 0.80 else COLOR_DANGER))
                child.setForeground(1, QtGui.QBrush(QtGui.QColor(acc_color)))
                child.setData(0, QtCore.Qt.UserRole, {
                    "type": "fold",
                    "key": key,
                    "fold_idx": f.fold_index,
                    "available": (f.fold_index in mr.fold_models),
                })
                top.addChild(child)

            # Por defecto, top-levels COLAPSADOS (vista de acordeón)
            top.setExpanded(False)

        # Mensajes de selección: deshabilitar botones hasta que el usuario
        # seleccione algo válido
        self.cv_ranking_selected_lbl.setText(
            "Selecciona un modelo del ranking para descargarlo o cargarlo."
        )
        self.btn_cv_download.setEnabled(False)
        self.btn_cv_apply.setEnabled(False)
        # «Descargar mejor modelo» se habilita en cuanto haya al menos un
        # modelo full entrenado.
        any_full = any(k in self._cv_full_models for k in self.last_cv_result.per_model)
        self.btn_cv_download_best.setEnabled(any_full)

    def _on_cv_ranking_selection(self):
        """
        Maneja la selección (posiblemente múltiple) en el QTreeWidget.

        Reglas:
            - Sin selección → botones deshabilitados.
            - 1 item → muestra detalle (full o fold concreto).
            - >1 items → deduplica por algoritmo, anuncia cuántos se cargarán.
        """
        items = [it for it in self.cv_ranking_tree.selectedItems()
                 if (it.data(0, QtCore.Qt.UserRole) or {}).get("available", False)]
        if not items:
            self.btn_cv_download.setEnabled(False)
            self.btn_cv_apply.setEnabled(False)
            self.cv_ranking_selected_lbl.setText(
                "Selecciona uno o varios modelos del ranking. "
                "Ctrl+click para añadir, Shift+click para rango."
            )
            return

        # Deduplicar por algoritmo (un solo modelo por clave). Preferencia:
        # un "full" gana sobre un "fold" del mismo algoritmo.
        per_key: Dict[str, Dict[str, Any]] = {}
        for it in items:
            sel = it.data(0, QtCore.Qt.UserRole)
            k = sel["key"]
            existing = per_key.get(k)
            if existing is None or (sel["type"] == "full" and existing["type"] != "full"):
                per_key[k] = sel

        self.btn_cv_download.setEnabled(True)
        self.btn_cv_apply.setEnabled(True)

        if len(per_key) == 1:
            # Una sola selección efectiva
            sel = next(iter(per_key.values()))
            key = sel["key"]
            display = MODEL_DISPLAY_NAMES.get(key, key)
            if sel["type"] == "full":
                descr = f"⭐  Mejor modelo de <b>{display}</b> (full dataset)"
            else:
                descr = f"<b>{display}</b> · fold #{sel['fold_idx']}"
            self.cv_ranking_selected_lbl.setText(
                f"<b>Seleccionado:</b> " + descr +
                " — listo para descargar o cargar al programa."
            )
        else:
            # Multi-selección
            names = " + ".join(
                MODEL_SHORT_NAMES.get(k, k) for k in per_key
            )
            self.cv_ranking_selected_lbl.setText(
                f"<b>Seleccionados ({len(per_key)} modelos):</b> "
                f"<span style='color:{COLOR_ACCENT};'>{names}</span> "
                f"— al cargar al programa se usarán todos juntos (voting ensemble)."
            )

    def _get_selected_model_bundles(self) -> List[Dict[str, Any]]:
        """
        Devuelve la lista de bundles {model, scaler, feature_names, ...} para
        los modelos actualmente seleccionados en el tree.

        Política de deduplicación: un solo bundle por algoritmo. Si el
        usuario seleccionó tanto el «full» como un fold del mismo algoritmo,
        se prioriza el «full».

        Cuando hay varios algoritmos seleccionados, se fuerza el uso del
        scaler «full» común (`_cv_full_scaler`). Esto es importante porque
        cada fold tiene su propio scaler (fit_transform sobre el train del
        fold para evitar leakage); mezclar modelos de distintos folds en
        un mismo MainWindow con un solo `mw.scaler` daría predicciones
        inconsistentes. Por eso:

            - Si en la selección múltiple aparece algún algoritmo en modo
              «fold» (no full) Y hay más de un algoritmo, lo sustituimos
              automáticamente por el «full» de ese algoritmo (si está
              disponible) y avisamos al usuario.
            - Si solo hay UN algoritmo seleccionado, respetamos la
              elección exacta del usuario (fold o full).
        """
        items = [it for it in self.cv_ranking_tree.selectedItems()
                 if (it.data(0, QtCore.Qt.UserRole) or {}).get("available", False)]
        if not items:
            return []

        # Deduplicar (full > fold del mismo algo)
        per_key: Dict[str, Dict[str, Any]] = {}
        for it in items:
            sel = it.data(0, QtCore.Qt.UserRole)
            k = sel["key"]
            existing = per_key.get(k)
            if existing is None or (sel["type"] == "full" and existing["type"] != "full"):
                per_key[k] = sel

        bundles: List[Dict[str, Any]] = []
        multi = len(per_key) > 1
        for key, sel in per_key.items():
            display = MODEL_DISPLAY_NAMES.get(key, key)
            # Si multi y el usuario eligió un fold → sustituir por full
            if multi and sel["type"] == "fold" and key in self._cv_full_models:
                bundles.append({
                    "key": key,
                    "model": self._cv_full_models[key],
                    "scaler": self._cv_full_scaler,
                    "feature_names": getattr(self, "_cv_feature_names", []),
                    "source_descr": f"Best of {display} (full dataset)",
                    "type": "full",
                })
                continue
            if sel["type"] == "full":
                if key not in self._cv_full_models:
                    continue
                bundles.append({
                    "key": key,
                    "model": self._cv_full_models[key],
                    "scaler": self._cv_full_scaler,
                    "feature_names": getattr(self, "_cv_feature_names", []),
                    "source_descr": f"Best of {display} (full dataset)",
                    "type": "full",
                })
            else:
                # fold (solo posible si es selección única)
                mr = self.last_cv_result.per_model[key]
                fi = sel["fold_idx"]
                if fi not in mr.fold_models:
                    continue
                bundles.append({
                    "key": key,
                    "model": mr.fold_models[fi],
                    "scaler": mr.fold_scalers[fi],
                    "feature_names": self.last_cv_result.feature_names,
                    "source_descr": f"{display} · fold #{fi}",
                    "type": "fold",
                    "fold_idx": fi,
                })
        return bundles

    # Wrapper retrocompat (algunos sitios viejos usaban el singular)
    def _get_selected_model_bundle(self) -> Optional[Dict[str, Any]]:
        bundles = self._get_selected_model_bundles()
        return bundles[0] if bundles else None

    def _on_cv_ranking_download(self):
        bundles = self._get_selected_model_bundles()
        if not bundles:
            return
        # Nombre por defecto
        if len(bundles) == 1:
            b = bundles[0]
            if b["type"] == "fold":
                default_name = f"cv_{b['key']}_fold{b['fold_idx']}.joblib"
            else:
                default_name = f"cv_best_{b['key']}.joblib"
        else:
            default_name = "cv_combined_" + "+".join(b["key"] for b in bundles) + ".joblib"

        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Descargar modelo(s)", default_name, "Joblib (*.joblib)"
        )
        if not path:
            return
        try:
            # Cuando hay multi-select, todos los bundles ya tienen el mismo
            # scaler (`_cv_full_scaler`) gracias a la lógica de
            # `_get_selected_model_bundles`. Si es un solo fold, usamos su
            # scaler particular.
            scaler = bundles[0]["scaler"]
            feature_names = bundles[0]["feature_names"]
            models_dict = {b["key"]: b["model"] for b in bundles}

            tr = TrainingResult(
                models=dict(models_dict),
                scaler=scaler,
                feature_names=feature_names,
                source="cv_ranking",
                n_samples=self.last_cv_result.config.target_total,
            )
            report_lines = []
            for b in bundles:
                key = b["key"]
                mr = self.last_cv_result.per_model[key]
                tr.metrics[key] = mr.mean_accuracy
                tr.cvs[key]     = mr.metric_array("accuracy")
                tr.cms[key]     = sum(
                    (f.confusion for f in mr.folds),
                    np.zeros((2, 2), dtype=int),
                )
                report_lines.append(
                    f"{b['source_descr']}: acc={mr.mean_accuracy:.4f} ± {mr.std_accuracy:.4f}"
                )
                tr.reports[key] = report_lines[-1]
            payload = {
                "format_version": MODEL_FORMAT_VERSION,
                "app_version":    APP_VERSION,
                "models":         models_dict,
                "rf":  models_dict.get("rf"),
                "svm": models_dict.get("svm"),
                "scaler":        scaler,
                "feature_names": feature_names,
                "metrics":       tr.to_dict(),
            }
            import joblib as _jl
            _jl.dump(payload, path)

            if len(bundles) == 1:
                self.mw.status.showMessage(
                    f"💾  Modelo descargado: {os.path.basename(path)}"
                )
                QtWidgets.QMessageBox.information(
                    self, "Modelo descargado",
                    f"<b>{bundles[0]['source_descr']}</b><br><br>"
                    f"Guardado en:<br>{path}"
                )
            else:
                names = " + ".join(
                    MODEL_SHORT_NAMES.get(b["key"], b["key"]) for b in bundles
                )
                self.mw.status.showMessage(
                    f"💾  Descargado combinado ({names}): {os.path.basename(path)}"
                )
                QtWidgets.QMessageBox.information(
                    self, "Modelos descargados",
                    f"<b>{names}</b> guardados en un único .joblib:<br><br>"
                    f"{path}<br><br>"
                    f"Al cargarlo, todos los modelos quedan disponibles y "
                    f"el voting ensemble los promedia automáticamente."
                )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error al descargar", str(e))

    def _on_cv_ranking_apply(self):
        """
        Carga los modelos seleccionados (uno o varios) al MainWindow.

        Importante: NO redirige a otra pestaña. El usuario decide cuándo
        ver los resultados — esto evita interrumpir su flujo de trabajo.
        """
        bundles = self._get_selected_model_bundles()
        if not bundles:
            return
        try:
            mw = self.mw
            # Sustituir TODOS los modelos del MW por los seleccionados.
            mw.models = {b["key"]: b["model"] for b in bundles}
            # Todos los bundles comparten el mismo scaler en multi-select
            # (gracias a la lógica de _get_selected_model_bundles).
            mw.scaler = bundles[0]["scaler"]
            mw.feature_names = bundles[0]["feature_names"]

            tr = TrainingResult(
                models=dict(mw.models),
                scaler=mw.scaler,
                feature_names=mw.feature_names,
                source="cv_ranking",
                n_samples=self.last_cv_result.config.target_total,
            )
            for b in bundles:
                key = b["key"]
                mr = self.last_cv_result.per_model[key]
                tr.metrics[key] = mr.mean_accuracy
                tr.cvs[key]     = mr.metric_array("accuracy")
                tr.cms[key]     = sum(
                    (f.confusion for f in mr.folds),
                    np.zeros((2, 2), dtype=int),
                )
                tr.reports[key] = (
                    f"{b['source_descr']}\n"
                    f"Criterio: {self.cv_ranking_criterion.currentText()}\n"
                    f"CV: {mr.mean_accuracy:.4f} ± {mr.std_accuracy:.4f}  "
                    f"·  F1 {mr.mean_f1:.4f}  ·  AUC {(mr.mean_roc_auc or 0):.4f}"
                )
            mw.last_training = tr
            mw.model_updated.emit()

            # Mensaje de estado distinto según número de modelos cargados
            if len(bundles) == 1:
                self.mw.status.showMessage(
                    f"📥  «{bundles[0]['source_descr']}» cargado al programa."
                )
            else:
                names = " + ".join(
                    MODEL_SHORT_NAMES.get(b["key"], b["key"]) for b in bundles
                )
                self.mw.status.showMessage(
                    f"📥  Cargados {len(bundles)} modelos: {names} "
                    f"(voting ensemble activo)."
                )
            # (v3.20) Sin redirección automática — el usuario navega
            # manualmente cuando quiera ver Análisis del modelo.
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error al cargar modelo", str(e))

    def _on_cv_download_best(self):
        """
        Diálogo de checkboxes para combinar varios «mejores modelos» en
        un único .joblib. Si el usuario marca RF + XGB, el archivo
        resultante contiene los dos modelos juntos (listos para ensemble).
        """
        if self.last_cv_result is None:
            return
        # Solo permitimos algoritmos para los que tengamos el modelo full
        avail = [k for k in MODEL_KEYS
                 if k in self._cv_full_models
                 and self._cv_full_models[k] is not None]
        if not avail:
            QtWidgets.QMessageBox.information(
                self, "Sin modelos completos",
                "Aún no hay modelos finales entrenados. Ejecuta una CV "
                "(KFold/StratifiedKFold) y vuelve a intentar."
            )
            return

        dlg = QtWidgets.QDialog(self)
        dlg.setWindowTitle("Descargar mejor modelo")
        dlg.setMinimumWidth(420)
        v = QtWidgets.QVBoxLayout(dlg)
        v.setContentsMargins(20, 16, 20, 12); v.setSpacing(10)

        v.addWidget(QtWidgets.QLabel(
            "<h3>Selecciona los algoritmos a incluir</h3>"
            "<p style='color:#aaa;'>Por cada algoritmo marcado se incluirá "
            "su <b>mejor modelo</b> (entrenado sobre todo el dataset). "
            "Si seleccionas varios, se guardan en un único .joblib junto "
            "con el scaler común, listos para usar como ensemble.</p>"
        ))

        # Tabla de checkboxes con métricas para que el usuario decida bien
        cbs: Dict[str, QtWidgets.QCheckBox] = {}
        algo_rows = self._compute_ranking_scores(
            self.last_cv_result,
            self.cv_ranking_criterion.currentText(),
        )
        # algo_rows ya está ordenado por el criterio actual; marcamos por
        # defecto el #1 para que sea fácil descargar el ganador
        first_avail_key = next((r["key"] for r in algo_rows if r["key"] in avail), None)
        for r in algo_rows:
            k = r["key"]
            if k not in avail:
                continue
            display = MODEL_DISPLAY_NAMES.get(k, k)
            cb = QtWidgets.QCheckBox(
                f"{display}  ·  acc={r['acc']:.4f}  ·  F1={r['f1']:.4f}"
                + (f"  ·  AUC={r['auc']:.4f}" if r['auc'] > 0 else "")
            )
            cb.setChecked(k == first_avail_key)
            cbs[k] = cb
            v.addWidget(cb)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        v.addWidget(btns)

        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return

        selected = [k for k, cb in cbs.items() if cb.isChecked()]
        if not selected:
            QtWidgets.QMessageBox.warning(
                self, "Sin selección",
                "No marcaste ningún algoritmo. No se descarga nada."
            )
            return

        # Nombre por defecto: concat de claves con '+'
        suffix = "+".join(selected)
        default_name = f"cv_best_{suffix}.joblib"
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Descargar mejor modelo combinado",
            default_name, "Joblib (*.joblib)"
        )
        if not path:
            return

        try:
            models_dict = {k: self._cv_full_models[k] for k in selected}
            # Para que `_on_load_model` legacy también funcione, exponemos
            # las claves rf/svm a nivel raíz si aplican.
            tr = TrainingResult(
                models=dict(models_dict),
                scaler=self._cv_full_scaler,
                feature_names=getattr(self, "_cv_feature_names", []),
                source="cv_ranking",
                n_samples=self.last_cv_result.config.target_total,
            )
            for k in selected:
                mr = self.last_cv_result.per_model[k]
                tr.metrics[k] = mr.mean_accuracy
                tr.cvs[k]     = mr.metric_array("accuracy")
                tr.cms[k]     = sum(
                    (f.confusion for f in mr.folds),
                    np.zeros((2, 2), dtype=int),
                )
                tr.reports[k] = (
                    f"Best of {MODEL_DISPLAY_NAMES.get(k, k)} (full dataset)\n"
                    f"CV: {mr.mean_accuracy:.4f} ± {mr.std_accuracy:.4f}"
                )
            payload = {
                "format_version": MODEL_FORMAT_VERSION,
                "app_version":    APP_VERSION,
                "models":         models_dict,
                "rf":  models_dict.get("rf"),
                "svm": models_dict.get("svm"),
                "scaler":        self._cv_full_scaler,
                "feature_names": getattr(self, "_cv_feature_names", []),
                "metrics":       tr.to_dict(),
            }
            import joblib as _jl
            _jl.dump(payload, path)
            display_names = " + ".join(
                MODEL_SHORT_NAMES.get(k, k) for k in selected
            )
            self.mw.status.showMessage(
                f"💾  Descargado: {os.path.basename(path)} ({display_names})"
            )
            QtWidgets.QMessageBox.information(
                self, "Modelo combinado descargado",
                f"<b>{display_names}</b> guardados en un único .joblib:<br>"
                f"{path}<br><br>"
                f"Al cargarlo en el programa, todos los modelos quedan "
                f"disponibles y el voting ensemble los promedia automáticamente."
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error al descargar", str(e))

    def _export_cv_csv(self, result: CrossValidationResult, path: str):
        """Exporta los resultados de CV a un CSV con secciones."""
        def fmt_opt(v: Optional[float], dec: int = 4) -> str:
            if v is None:
                return ""
            return f"{v:.{dec}f}"

        with open(path, "w", encoding="utf-8") as fh:
            fh.write(f"# HydroAnalyzer CV results · timestamp={result.timestamp}\n")
            fh.write(f"# strategy={result.config.strategy} "
                      f"n_splits={result.config.n_splits} "
                      f"calibrate={result.config.calibrate}\n")
            fh.write(f"# total_time_s={result.total_time_s:.2f}\n\n")

            # Sección 1: resumen por modelo
            fh.write("# === RESUMEN POR MODELO ===\n")
            fh.write("modelo,folds,acc_mean,acc_std,prec_mean,recall_mean,"
                      "f1_mean,auc_mean,total_time_s\n")
            for mr in result.per_model.values():
                fh.write(
                    f"{mr.model_key},{len(mr.folds)},"
                    f"{mr.mean_accuracy:.4f},{mr.std_accuracy:.4f},"
                    f"{mr.mean_precision:.4f},{mr.mean_recall:.4f},"
                    f"{mr.mean_f1:.4f},"
                    f"{fmt_opt(mr.mean_roc_auc)},"
                    f"{mr.total_time_s:.2f}\n"
                )
            fh.write("\n")

            # Sección 2: detalle por fold
            fh.write("# === DETALLE POR FOLD ===\n")
            fh.write("modelo,fold,acc,prec,recall,f1,auc,n_train,n_val,time_s\n")
            for mr in result.per_model.values():
                for f in mr.folds:
                    fh.write(
                        f"{mr.model_key},{f.fold_index},"
                        f"{f.accuracy:.4f},{f.precision:.4f},"
                        f"{f.recall:.4f},{f.f1:.4f},"
                        f"{fmt_opt(f.roc_auc)},"
                        f"{f.n_train},{f.n_val},{f.train_time_s:.2f}\n"
                    )
            fh.write("\n")

            # Sección 3: muestras conflictivas / outliers
            fh.write("# === ANÁLISIS POR MUESTRA ===\n")
            stats = self._compute_sample_stats(result)
            stats.sort(key=lambda s: s["reliability"])
            fh.write("indice,nombre,clase_real,n_aciertos,n_errores,"
                      "confiabilidad_pct,prob_bypass_mean,prob_bypass_std,"
                      "modelos_que_fallaron\n")
            labels = {0: "Normal", 1: "Bypass"}
            for s in stats:
                fh.write(
                    f'{s["index"]},'
                    f'"{s["name"]}",'
                    f'{labels[s["true"]]},'
                    f'{s["n_correct"]},{s["n_wrong"]},'
                    f'{s["reliability"]*100:.2f},'
                    f'{fmt_opt(s.get("mean_prob_bypass"))},'
                    f'{fmt_opt(s.get("std_prob_bypass"))},'
                    f'"{",".join(s["wrong_list"])}"\n'
                )

    # ============================================================
    # SUB-PESTAÑA: ENTRENAMIENTO (UI original)
    # ============================================================
    def _build_training_ui(self, container: QtWidgets.QWidget):
        """UI clásica de entrenamiento — recibe el QWidget contenedor."""
        root = QtWidgets.QHBoxLayout(container)
        root.setContentsMargins(10, 10, 10, 10); root.setSpacing(10)

        # === Panel izquierdo: listas de archivos ==========================
        left = QtWidgets.QWidget(); ll = QtWidgets.QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)

        grp_files = QtWidgets.QGroupBox("📂  Carga de datos reales")
        gf = QtWidgets.QVBoxLayout(grp_files)

        btn_row = QtWidgets.QHBoxLayout()
        b_no  = QtWidgets.QPushButton("＋ Normal (clase 0)")
        b_yes = QtWidgets.QPushButton("＋ Bypass (clase 1)")
        b_no.clicked.connect(self.load_files_no)
        b_yes.clicked.connect(self.load_files_yes)
        btn_row.addWidget(b_no); btn_row.addWidget(b_yes)
        gf.addLayout(btn_row)

        self.lbl_counts = QtWidgets.QLabel("Archivos: 0 Normal · 0 Bypass")
        self.lbl_counts.setAlignment(QtCore.Qt.AlignCenter)
        self.lbl_counts.setObjectName("countsLabel")
        gf.addWidget(self.lbl_counts)

        # Botón para filtrar todas las señales cargadas usando la
        # configuración de la pestaña Simulador. Modifica permanentemente
        # data_no / data_yes — el siguiente entrenamiento usa la versión
        # filtrada.
        self.btn_filter_all = QtWidgets.QPushButton(
            "🧹  Filtrar señales cargadas (usa filtros del Simulador)"
        )
        self.btn_filter_all.setToolTip(
            "Aplica la configuración actual de filtros del Simulador a TODAS\n"
            "las señales cargadas en este entrenador. Útil para limpiar\n"
            "spikes/outliers antes de entrenar.\n\n"
            "Importante: configura primero los filtros en la pestaña\n"
            "«Simulador» (probando con UNA señal). Cuando estés satisfecho,\n"
            "vuelve aquí y pulsa este botón para aplicarlos a todas."
        )
        self.btn_filter_all.clicked.connect(self.on_filter_all_signals)
        gf.addWidget(self.btn_filter_all)

        gf.addWidget(QtWidgets.QLabel("Clase 0 — Normal:"))
        self.list_no = QtWidgets.QListWidget()
        self.list_no.itemSelectionChanged.connect(
            lambda: self._preview(self.list_no, self.data_no, COLOR_CYAN, "Normal")
        )
        gf.addWidget(self.list_no)

        row1 = QtWidgets.QHBoxLayout()
        br1 = QtWidgets.QPushButton("🗑  Quitar"); br1.clicked.connect(
            lambda: self._remove_selected(self.list_no, self.data_no))
        bc1 = QtWidgets.QPushButton("Limpiar"); bc1.clicked.connect(
            lambda: self._clear(self.list_no, self.data_no))
        row1.addWidget(br1); row1.addWidget(bc1); gf.addLayout(row1)

        gf.addWidget(QtWidgets.QLabel("Clase 1 — Bypass:"))
        self.list_yes = QtWidgets.QListWidget()
        self.list_yes.itemSelectionChanged.connect(
            lambda: self._preview(self.list_yes, self.data_yes, COLOR_DANGER, "Bypass")
        )
        gf.addWidget(self.list_yes)

        row2 = QtWidgets.QHBoxLayout()
        br2 = QtWidgets.QPushButton("🗑  Quitar"); br2.clicked.connect(
            lambda: self._remove_selected(self.list_yes, self.data_yes))
        bc2 = QtWidgets.QPushButton("Limpiar"); bc2.clicked.connect(
            lambda: self._clear(self.list_yes, self.data_yes))
        row2.addWidget(br2); row2.addWidget(bc2); gf.addLayout(row2)

        ll.addWidget(grp_files)

        # === Panel central: preview =======================================
        center = QtWidgets.QWidget(); cl = QtWidgets.QVBoxLayout(center)
        cl.setContentsMargins(0, 0, 0, 0)
        grp_prev = QtWidgets.QGroupBox("📈  Previsualización (clic en la lista)")
        gp = QtWidgets.QVBoxLayout(grp_prev)
        self.canvas = PlotCanvas(nrows=1, ncols=1)
        gp.addWidget(NavigationToolbar(self.canvas, self))
        gp.addWidget(self.canvas)
        cl.addWidget(grp_prev)
        self.canvas.show_empty("Selecciona un archivo de las listas")

        # === Panel derecho: parámetros y entrenamiento ====================
        right = QtWidgets.QWidget(); rl = QtWidgets.QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)

        grp_aug = QtWidgets.QGroupBox("🔀  Data augmentation")
        fa = QtWidgets.QFormLayout(grp_aug)
        self.n_aug = QtWidgets.QSpinBox(); self.n_aug.setRange(0, 50); self.n_aug.setValue(6)
        self.n_aug.setToolTip("Aumentaciones por señal original (ruido, amplitud, shift, stretch).")
        self.target_total = QtWidgets.QSpinBox(); self.target_total.setRange(20, 50000); self.target_total.setValue(1200)
        self.target_total.setToolTip("Tamaño final del dataset (balanceado 50/50).")
        fa.addRow("Aumentaciones:", self.n_aug)
        fa.addRow("Dataset final:", self.target_total)

        # Botones de sugerencia / reset para Augmentation
        aug_btn_row = QtWidgets.QHBoxLayout()
        aug_btn_row.setSpacing(6)
        self.btn_suggest_aug = QtWidgets.QPushButton("🪄  Activar valores sugeridos")
        self.btn_suggest_aug.setToolTip(
            "Analiza las señales cargadas y propone valores adaptados\n"
            "para n_aug y dataset final. La heurística considera el\n"
            "número de señales por clase, la frecuencia de muestreo y\n"
            "el balance, buscando el modelo más robusto posible."
        )
        self.btn_suggest_aug.clicked.connect(self._on_suggest_aug)
        self.btn_reset_aug = QtWidgets.QPushButton("↺  Restaurar")
        self.btn_reset_aug.setToolTip(
            "Restaura los valores por defecto del bloque de augmentation."
        )
        self.btn_reset_aug.clicked.connect(self._on_reset_aug)
        aug_btn_row.addWidget(self.btn_suggest_aug)
        aug_btn_row.addWidget(self.btn_reset_aug)
        fa.addRow(aug_btn_row)

        rl.addWidget(grp_aug)

        grp_ml = QtWidgets.QGroupBox("🧠  Parámetros ML")
        fm = QtWidgets.QFormLayout(grp_ml)

        self.use_rf   = self._make_model_checkbox("rf",   "Random Forest", default=True)
        self.use_svm  = self._make_model_checkbox("svm",  "SVM (RBF)",     default=True)
        self.use_xgb  = self._make_model_checkbox("xgb",  "XGBoost",       default=False)
        self.use_lgbm = self._make_model_checkbox("lgbm", "LightGBM",      default=False)

        self.n_estimators = QtWidgets.QSpinBox(); self.n_estimators.setRange(10, 1000); self.n_estimators.setValue(150)
        self.n_estimators.setToolTip("Aplicado a RF, XGBoost y LightGBM.")
        self.svm_c    = QtWidgets.QDoubleSpinBox(); self.svm_c.setRange(0.01, 100.0); self.svm_c.setSingleStep(0.1); self.svm_c.setValue(1.0)
        self.test_size = QtWidgets.QDoubleSpinBox(); self.test_size.setRange(0.05, 0.5); self.test_size.setSingleStep(0.05); self.test_size.setValue(0.25)
        self.lr_boost = QtWidgets.QDoubleSpinBox(); self.lr_boost.setRange(0.001, 1.0); self.lr_boost.setSingleStep(0.01); self.lr_boost.setDecimals(3); self.lr_boost.setValue(0.1)
        self.lr_boost.setToolTip("Tasa de aprendizaje para XGBoost / LightGBM.")
        self.max_depth = QtWidgets.QSpinBox(); self.max_depth.setRange(2, 30); self.max_depth.setValue(6)
        self.max_depth.setToolTip("Profundidad máxima para XGBoost / LightGBM.")

        fm.addRow("", self.use_rf)
        fm.addRow("", self.use_svm)
        fm.addRow("", self.use_xgb)
        fm.addRow("", self.use_lgbm)
        fm.addRow("Árboles (RF/XGB/LGBM):", self.n_estimators)
        fm.addRow("SVM C:", self.svm_c)
        fm.addRow("Tasa aprendizaje:", self.lr_boost)
        fm.addRow("Max depth:", self.max_depth)
        fm.addRow("Test size:", self.test_size)

        self.calibrate = QtWidgets.QCheckBox(
            "Calibrar probabilidades (isotonic, recomendado)"
        )
        self.calibrate.setChecked(True)
        self.calibrate.setToolTip(
            "Envuelve cada modelo en CalibratedClassifierCV(method='isotonic').\n"
            "Hace que los % de confianza sean realistas — un modelo que muestra\n"
            "90% acertará realmente cerca del 90% de las veces."
        )
        fm.addRow("", self.calibrate)

        # Botones de sugerencia / reset para Parámetros ML
        ml_btn_row = QtWidgets.QHBoxLayout()
        ml_btn_row.setSpacing(6)
        self.btn_suggest_ml = QtWidgets.QPushButton("🪄  Activar valores sugeridos")
        self.btn_suggest_ml.setToolTip(
            "Sugiere parámetros ML adaptados al dataset cargado:\n"
            "  • Árboles RF según el número de señales y la fs.\n"
            "  • SVM C ajustado al tamaño del dataset.\n"
            "  • Test size mayor para datasets pequeños (mejor validación).\n"
            "  • Activa/desactiva SVM si el dataset es muy pequeño."
        )
        self.btn_suggest_ml.clicked.connect(self._on_suggest_ml)
        self.btn_reset_ml = QtWidgets.QPushButton("↺  Restaurar")
        self.btn_reset_ml.setToolTip(
            "Restaura los valores por defecto del bloque de Parámetros ML."
        )
        self.btn_reset_ml.clicked.connect(self._on_reset_ml)
        ml_btn_row.addWidget(self.btn_suggest_ml)
        ml_btn_row.addWidget(self.btn_reset_ml)
        fm.addRow(ml_btn_row)

        rl.addWidget(grp_ml)

        self.btn_train = QtWidgets.QPushButton("🚀  Aumentar y entrenar")
        self.btn_train.setObjectName("primaryButton"); self.btn_train.setMinimumHeight(40)
        self.btn_train.clicked.connect(self.on_train)
        rl.addWidget(self.btn_train)

        # (v4.2) El antiguo checkbox «Activar Validación Cruzada» se
        # eliminó: la sub-pestaña de Validación Cruzada ahora está
        # siempre disponible junto a «Entrenamiento».

        self.summary = QtWidgets.QPlainTextEdit(); self.summary.setReadOnly(True)
        self.summary.setPlainText(
            "Flujo:\n"
            "  1. Carga CSVs para cada clase.\n"
            "  2. Ajusta augmentation y parámetros ML.\n"
            "  3. Entrena — las métricas aparecen\n"
            "     en la pestaña «Análisis del modelo».\n\n"
            "Formatos aceptados: columnas t,p | time,pressure\n"
            "o dos primeras columnas numéricas sin cabecera."
        )
        rl.addWidget(self.summary, 1)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(left); splitter.addWidget(center); splitter.addWidget(right)
        splitter.setSizes([320, 620, 360])
        splitter.setStretchFactor(1, 1)
        root.addWidget(splitter)

    # ---------- archivo handling ----------
    def load_files_no(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Señales normales (CSV / TXT / LOG)", "", SIGNAL_FILE_FILTER)
        self._process(files, self.data_no, self.list_no, 0)

    def load_files_yes(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Señales bypass (CSV / TXT / LOG)", "", SIGNAL_FILE_FILTER)
        self._process(files, self.data_yes, self.list_yes, 1)

    def _process(self, files, target, ui_list, cls):
        added = 0
        for f in files:
            try:
                t, p = load_csv_signal(f)
                target.append((f, t, p, infer_fs(t)))
                ui_list.addItem(os.path.basename(f))
                added += 1
            except Exception as e:
                QtWidgets.QMessageBox.warning(self, "Error",
                                              f"{os.path.basename(f)}:\n{e}")
        self._update_counts()
        self.mw.status.showMessage(f"✅  {added} archivo(s) añadido(s) a clase {cls}.")

    def _remove_selected(self, ui_list, data):
        rows = sorted({i.row() for i in ui_list.selectedIndexes()}, reverse=True)
        for r in rows:
            ui_list.takeItem(r)
            del data[r]
        self._update_counts()

    def _clear(self, ui_list, data):
        ui_list.clear()
        data.clear()
        self._update_counts()
        self.canvas.show_empty("Selecciona un archivo de las listas")

    def on_filter_all_signals(self):
        """
        Aplica la configuración de filtros del Simulador a TODAS las
        señales cargadas en este entrenador.

        Workflow esperado por el usuario:
            1. Va a la pestaña «Simulador».
            2. Carga UNA señal de muestra y configura los filtros que
               quiera (intervalo manual, IQR, Hampel, etc.) — viendo
               el efecto en vivo.
            3. Vuelve a esta pestaña y pulsa este botón.
            4. Cada señal de data_no / data_yes se filtra con esa misma
               configuración, sustituyendo la versión original en memoria.
        """
        if not self.data_no and not self.data_yes:
            QtWidgets.QMessageBox.information(
                self, "Filtrar señales",
                "No hay señales cargadas todavía.\n"
                "Carga al menos un archivo en alguna de las dos clases\n"
                "antes de filtrar."
            )
            return

        # Tomar la configuración tal cual del Simulador
        try:
            sim = self.mw.tab_sim
            cfg = sim._build_filter_config_from_ui()
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Filtrar señales",
                f"No se pudo leer la configuración de filtros del Simulador:\n{e}"
            )
            return

        if not cfg.enabled:
            QtWidgets.QMessageBox.warning(
                self, "Filtrado desactivado",
                "El filtrado global está DESACTIVADO en la pestaña Simulador.\n"
                "Ve allá, marca «Activar filtrado» y configura las etapas\n"
                "que quieras antes de volver a pulsar este botón."
            )
            return

        # Cuántas etapas reales se aplicarán (para informar al usuario)
        active_stages = [
            name for name, on in [
                ("Diferencia con vecinos", cfg.neighbor_enabled),
                ("Hampel",                 cfg.hampel_enabled),
                ("IQR",                    cfg.iqr_enabled),
                ("Duración",               cfg.duration_enabled),
                ("Manual por intervalo",   cfg.manual_enabled),
                ("Pasa-bajos",             cfg.lowpass_enabled),
            ] if on
        ]
        if not active_stages:
            QtWidgets.QMessageBox.warning(
                self, "Sin filtros activos",
                "El filtrado global está activo, pero ninguna etapa concreta\n"
                "lo está. Configura al menos una etapa en el Simulador."
            )
            return

        # Confirmación
        total = len(self.data_no) + len(self.data_yes)
        ret = QtWidgets.QMessageBox.question(
            self, "Confirmar filtrado",
            f"Se aplicarán los siguientes filtros del Simulador a "
            f"<b>{total}</b> señal(es) cargada(s):<br><br>"
            "• " + "<br>• ".join(active_stages) +
            "<br><br>El cambio es permanente: las señales originales se "
            "reemplazan en memoria por su versión filtrada. "
            "Si quieres volver al estado anterior, tendrás que recargar los CSVs.<br><br>"
            "¿Continuar?",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if ret != QtWidgets.QMessageBox.Yes:
            return

        # Aplicar
        n_total_outliers = 0
        n_total_failed = 0

        def filter_group(group: list) -> Tuple[int, int]:
            """Filtra in-place y devuelve (sum_outliers, n_failed)."""
            sum_out = 0
            n_fail = 0
            for i, (path, t, p, fs) in enumerate(group):
                try:
                    p_filt, diag = apply_filter_pipeline(p, fs, cfg, t)
                    group[i] = (path, t, p_filt, fs)
                    # Contar outliers totales detectados (campo combined_outliers
                    # cuando exista; si no, sumar los stages)
                    if "combined_outliers" in diag:
                        sum_out += int(np.sum(diag["combined_outliers"]))
                    else:
                        for k, v in diag.items():
                            if hasattr(v, "sum"):
                                try:
                                    sum_out += int(np.sum(v))
                                except Exception:
                                    pass
                except Exception:
                    n_fail += 1
            return sum_out, n_fail

        out_no, fail_no = filter_group(self.data_no)
        out_yes, fail_yes = filter_group(self.data_yes)
        n_total_outliers = out_no + out_yes
        n_total_failed   = fail_no + fail_yes

        # Refrescar la previsualización si había una señal mostrada
        # (su data fue reemplazada in-place; el nuevo preview lo reflejará)
        if self.list_no.currentItem() is not None:
            self._preview(self.list_no, self.data_no, COLOR_CYAN, "Normal")
        elif self.list_yes.currentItem() is not None:
            self._preview(self.list_yes, self.data_yes, COLOR_DANGER, "Bypass")

        # Mensaje resumen
        msg_lines = [
            f"<b>Filtrado completado.</b>",
            f"",
            f"Señales procesadas: {total}",
            f"&nbsp;&nbsp;• Normal: {len(self.data_no)}  "
              f"({out_no} muestras suprimidas)",
            f"&nbsp;&nbsp;• Bypass: {len(self.data_yes)}  "
              f"({out_yes} muestras suprimidas)",
            f"<br><b>Total de muestras suprimidas:</b> {n_total_outliers}",
        ]
        if n_total_failed > 0:
            msg_lines.append(
                f"<br><span style='color:{COLOR_WARNING};'>"
                f"⚠ {n_total_failed} señal(es) fallaron al filtrar y "
                f"se mantuvieron sin cambios.</span>"
            )
        QtWidgets.QMessageBox.information(
            self, "Filtrado completado", "<br>".join(msg_lines)
        )
        self.mw.status.showMessage(
            f"🧹  Filtradas {total} señales · {n_total_outliers} muestras suprimidas."
        )

    def _update_counts(self):
        self.lbl_counts.setText(
            f"Archivos: {len(self.data_no)} Normal · {len(self.data_yes)} Bypass"
        )
        # Refrescar también el resumen del panel CV (si existe ya creado)
        if hasattr(self, "cv_data_status"):
            self._refresh_cv_data_status()

    # ---------- sugerencias / reset de parámetros ----------
    def _gather_dataset_stats(self) -> Optional[Dict[str, Any]]:
        """Calcula n por clase, lista de fs y duraciones del dataset cargado."""
        if not self.data_no and not self.data_yes:
            QtWidgets.QMessageBox.information(
                self, "Sugerencias",
                "Primero carga señales en al menos una de las dos clases\n"
                "(Normal y/o Bypass) para que el sistema pueda analizarlas."
            )
            return None
        fs_list: List[int] = []
        dur_list: List[float] = []
        for grp in (self.data_no, self.data_yes):
            for (path, t, p, fs) in grp:
                fs_list.append(int(fs))
                if len(t) >= 2:
                    dur_list.append(float(t[-1] - t[0]))
        return {
            "n_normal": len(self.data_no),
            "n_bypass": len(self.data_yes),
            "fs_list": fs_list,
            "duration_list": dur_list,
        }

    def _on_suggest_aug(self):
        stats = self._gather_dataset_stats()
        if stats is None:
            return
        sugg = suggest_training_params(**stats)
        # Bloquear señales no es estrictamente necesario aquí (no hay
        # callbacks reactivos), pero por consistencia.
        self.n_aug.blockSignals(True)
        self.target_total.blockSignals(True)
        try:
            self.n_aug.setValue(int(sugg["n_aug"]))
            self.target_total.setValue(int(sugg["target_total"]))
        finally:
            self.n_aug.blockSignals(False)
            self.target_total.blockSignals(False)

        self.summary.setPlainText(
            "🪄 Valores sugeridos aplicados a Augmentation:\n\n"
            f"  • Aumentaciones por señal: {sugg['n_aug']}\n"
            f"  • Dataset final         : {sugg['target_total']}\n\n"
            "Razonamiento:\n" + sugg["reasoning"]
        )
        self.mw.status.showMessage(
            f"🪄 Augmentation sugerida: n_aug={sugg['n_aug']}, "
            f"target_total={sugg['target_total']}"
        )

    def _on_reset_aug(self):
        self.n_aug.setValue(6)
        self.target_total.setValue(1200)
        self.mw.status.showMessage("↺ Augmentation restaurada a valores por defecto.")

    def _on_suggest_ml(self):
        stats = self._gather_dataset_stats()
        if stats is None:
            return
        sugg = suggest_training_params(**stats)
        self.n_estimators.blockSignals(True)
        self.svm_c.blockSignals(True)
        self.test_size.blockSignals(True)
        self.use_rf.blockSignals(True)
        self.use_svm.blockSignals(True)
        try:
            self.n_estimators.setValue(int(sugg["n_estimators"]))
            self.svm_c.setValue(float(sugg["svm_c"]))
            self.test_size.setValue(float(sugg["test_size"]))
            self.use_rf.setChecked(bool(sugg["use_rf"]))
            self.use_svm.setChecked(bool(sugg["use_svm"]))
        finally:
            self.n_estimators.blockSignals(False)
            self.svm_c.blockSignals(False)
            self.test_size.blockSignals(False)
            self.use_rf.blockSignals(False)
            self.use_svm.blockSignals(False)

        self.summary.setPlainText(
            "🪄 Valores sugeridos aplicados a Parámetros ML:\n\n"
            f"  • Random Forest activo : {sugg['use_rf']}\n"
            f"  • SVM activo           : {sugg['use_svm']}\n"
            f"  • Árboles RF           : {sugg['n_estimators']}\n"
            f"  • SVM C                : {sugg['svm_c']}\n"
            f"  • Test size            : {sugg['test_size']}\n\n"
            "Razonamiento:\n" + sugg["reasoning"]
        )
        self.mw.status.showMessage(
            f"🪄 ML sugerido: árboles={sugg['n_estimators']}, "
            f"C={sugg['svm_c']}, test={sugg['test_size']}"
        )

    def _on_reset_ml(self):
        self.use_rf.setChecked(True)
        self.use_svm.setChecked(True)
        self.n_estimators.setValue(150)
        self.svm_c.setValue(1.0)
        self.test_size.setValue(0.25)
        self.mw.status.showMessage("↺ Parámetros ML restaurados a valores por defecto.")

    def _preview(self, ui_list, data, color, cls_label):
        row = ui_list.currentRow()
        if row < 0 or row >= len(data):
            return
        path, t, p, fs = data[row]
        self.canvas.clear_axes()
        ax = self.canvas.axes[0, 0]
        ax.plot(t, p, color=color, linewidth=1.2)
        ax.set_title(f"[{cls_label}] {os.path.basename(path)}   fs={fs} Hz",
                     fontsize=10, pad=6)
        ax.set_xlabel("Tiempo (s)"); ax.set_ylabel("Presión")
        self.canvas.draw_idle()

    # ---------- entrenamiento ----------
    def _make_model_checkbox(self, key: str, label: str,
                              default: bool = False) -> QtWidgets.QCheckBox:
        """Crea un checkbox para activar un modelo. Si la dependencia
        no está instalada (XGB/LGBM), lo deshabilita y explica por qué."""
        cb = QtWidgets.QCheckBox(label)
        if model_is_available(key):
            cb.setChecked(default)
            cb.setEnabled(True)
        else:
            cb.setChecked(False)
            cb.setEnabled(False)
            pkg = "xgboost" if key == "xgb" else "lightgbm"
            cb.setToolTip(
                f"Esta opción requiere la librería '{pkg}'.\n"
                f"Instálala con:  pip install {pkg}"
            )
            cb.setText(f"{label}  (no instalado)")
        return cb

    def _selected_models(self) -> List[str]:
        """Lista de claves de modelos seleccionados en la UI."""
        out = []
        if self.use_rf.isChecked():   out.append("rf")
        if self.use_svm.isChecked():  out.append("svm")
        if self.use_xgb.isChecked():  out.append("xgb")
        if self.use_lgbm.isChecked(): out.append("lgbm")
        return out

    def on_train(self):
        if not self.data_no or not self.data_yes:
            QtWidgets.QMessageBox.warning(self, "Aviso",
                                          "Necesitas al menos un archivo en cada clase.")
            return
        keys = self._selected_models()
        if not keys:
            QtWidgets.QMessageBox.warning(self, "Aviso",
                                          "Selecciona al menos un modelo.")
            return

        worker = RealTrainingWorker(
            data_no=self.data_no, data_yes=self.data_yes,
            n_aug=self.n_aug.value(), target_total=self.target_total.value(),
            n_estimators=self.n_estimators.value(),
            svm_c=self.svm_c.value(), test_size=self.test_size.value(),
            models_to_train=keys,
            calibrate=self.calibrate.isChecked(),
            learning_rate=self.lr_boost.value(),
            max_depth=self.max_depth.value(),
        )
        self.mw.start_training(worker)


class ModelAnalysisTab(QtWidgets.QWidget):
    """Pestaña 4 — métricas, importancia, matrices de confusión."""
    def __init__(self, main_window):
        super().__init__()
        self.mw = main_window
        self._build_ui()
        self.refresh()

    def _build_ui(self):
        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10); root.setSpacing(8)

        # ---- info banner (sin botones de re-eval ya, viven en sub-pestañas)
        top = QtWidgets.QFrame(); top.setObjectName("infoFrame")
        tl = QtWidgets.QHBoxLayout(top)
        tl.setContentsMargins(14, 10, 14, 10)

        self.info_label = QtWidgets.QLabel("Sin modelo cargado.")
        self.info_label.setObjectName("infoLabel")
        tl.addWidget(self.info_label, 1)
        root.addWidget(top)

        # ---- pestañas de visualización -----------------------------------
        self.viz_tabs = QtWidgets.QTabWidget()
        self.viz_tabs.setDocumentMode(True)

        # Reporte texto
        self.report_text = QtWidgets.QPlainTextEdit()
        self.report_text.setReadOnly(True)
        w1 = QtWidgets.QWidget(); l1 = QtWidgets.QVBoxLayout(w1)
        l1.setContentsMargins(0, 0, 0, 0); l1.addWidget(self.report_text)
        self.viz_tabs.addTab(w1, "📋  Métricas")

        # Feature importance
        self.c_feat = PlotCanvas(nrows=1, ncols=1)
        w2 = QtWidgets.QWidget(); l2 = QtWidgets.QVBoxLayout(w2)
        l2.setContentsMargins(0, 0, 0, 0)
        l2.addWidget(NavigationToolbar(self.c_feat, self))
        l2.addWidget(self.c_feat)
        self.viz_tabs.addTab(w2, "🏆  Importancia")

        # ─── Pestaña «Matriz de confusión» con 2 sub-secciones ───────
        # Sub-pestañas mutuamente excluyentes (sólo se ve una a la vez)
        # para que la matriz visual tenga TODO el ancho disponible cuando
        # hay 4 modelos + ENS (5 paneles) y no quede comprimida.
        cm_subtabs = QtWidgets.QTabWidget()
        cm_subtabs.setDocumentMode(True)

        # ── Sub-sección A: Matriz visual ──────────────────────────────
        cm_visual = QtWidgets.QWidget()
        cm_visual_lay = QtWidgets.QVBoxLayout(cm_visual)
        cm_visual_lay.setContentsMargins(0, 0, 0, 0); cm_visual_lay.setSpacing(4)
        self.c_cm = PlotCanvas(nrows=1, ncols=2)
        cm_visual_lay.addWidget(NavigationToolbar(self.c_cm, self))
        cm_visual_lay.addWidget(self.c_cm)
        cm_subtabs.addTab(cm_visual, "🎯  Matriz visual")

        # ── Sub-sección B: Detalle por señal ──────────────────────────
        cm_detail = QtWidgets.QWidget()
        cm_detail_lay = QtWidgets.QVBoxLayout(cm_detail)
        cm_detail_lay.setContentsMargins(8, 4, 8, 4); cm_detail_lay.setSpacing(6)

        # Cabecera de la tabla con título + filtros + exportar
        det_header = QtWidgets.QHBoxLayout()
        det_header.setSpacing(6)
        det_title = QtWidgets.QLabel("📋  Detalle por señal")
        det_title.setStyleSheet(
            f"color:{COLOR_ACCENT}; font-weight:700; font-size:11pt;"
        )
        det_header.addWidget(det_title)
        det_header.addStretch(1)

        self.cm_filter_combo = QtWidgets.QComboBox()
        # Los items se reconfiguran dinámicamente en _reconfigure_filter_combo
        self.cm_filter_combo.addItems([
            "Todas",
            "Solo errores (cualquier modelo)",
            "Solo aciertos (todos)",
            "Solo clase Normal",
            "Solo clase Bypass",
        ])
        self.cm_filter_combo.setToolTip(
            "Filtrar la tabla por tipo de fila."
        )
        self.cm_filter_combo.currentIndexChanged.connect(
            lambda *_: self._apply_per_sample_filter()
        )
        det_header.addWidget(QtWidgets.QLabel("Mostrar:"))
        det_header.addWidget(self.cm_filter_combo)

        self.btn_cm_export = QtWidgets.QPushButton("⬇  Exportar CSV")
        self.btn_cm_export.setToolTip(
            "Exporta la tabla actual (con el filtro aplicado) a un archivo CSV."
        )
        self.btn_cm_export.clicked.connect(self._on_export_per_sample)
        det_header.addWidget(self.btn_cm_export)

        cm_detail_lay.addLayout(det_header)

        # Resumen rápido (n_total, n_errores por modelo)
        self.lbl_per_sample_summary = QtWidgets.QLabel("Sin datos.")
        self.lbl_per_sample_summary.setStyleSheet(
            f"color:{COLOR_TEXT_DIM}; font-size:9pt; padding:2px 4px;"
        )
        self.lbl_per_sample_summary.setWordWrap(True)
        cm_detail_lay.addWidget(self.lbl_per_sample_summary)

        # Tabla — las columnas se generan dinámicamente en _render_per_sample
        # según los modelos disponibles. Aquí solo creamos el widget.
        self.tbl_per_sample = QtWidgets.QTableWidget(0, 0)
        self.tbl_per_sample.setEditTriggers(
            QtWidgets.QAbstractItemView.NoEditTriggers
        )
        self.tbl_per_sample.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectRows
        )
        self.tbl_per_sample.setSortingEnabled(True)
        self.tbl_per_sample.setAlternatingRowColors(True)
        self.tbl_per_sample.verticalHeader().setVisible(False)
        self.tbl_per_sample.setToolTip(
            "Cada fila = una señal evaluada.\n"
            "Click en una cabecera para ordenar.\n"
            "Las filas erróneas se resaltan en rojo claro."
        )
        cm_detail_lay.addWidget(self.tbl_per_sample, 1)
        cm_subtabs.addTab(cm_detail, "📋  Detalle por señal")

        w3 = QtWidgets.QWidget(); l3 = QtWidgets.QVBoxLayout(w3)
        l3.setContentsMargins(0, 0, 0, 0)
        l3.addWidget(cm_subtabs)
        self.viz_tabs.addTab(w3, "🎯  Matriz de confusión")

        # === Sub-pestaña 4: Re-eval sintético ============================
        self.viz_tabs.addTab(self._build_reeval_synthetic_tab(),
                              "🧪  Re-eval sintético")

        # === Sub-pestaña 5: Re-eval experimental =========================
        self.viz_tabs.addTab(self._build_reeval_experimental_tab(),
                              "📂  Re-eval experimental")

        # Cuando el usuario cambia a la pestaña experimental, refrescamos
        # el indicador de "datos disponibles" porque pueden haber cambiado
        # mientras tanto (ej. acabar de cargar señales en otra pestaña).
        self.viz_tabs.currentChanged.connect(self._on_viz_tab_changed)

        root.addWidget(self.viz_tabs, 1)

    @QtCore.pyqtSlot(int)
    def _on_viz_tab_changed(self, idx: int):
        """Mantiene el estado de la sub-pestaña experimental actualizado."""
        self._update_reeval_exp_status()

    # ------------------------------------------------------------------
    def _build_reeval_synthetic_tab(self) -> QtWidgets.QWidget:
        """Sub-pestaña con controles para re-evaluación sintética."""
        page = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(page)
        outer.setContentsMargins(10, 10, 10, 10); outer.setSpacing(10)

        # — Lado izquierdo: parámetros (con scroll) —
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True); scroll.setFrameShape(QtWidgets.QFrame.NoFrame)
        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0)
        scroll.setWidget(left)

        info = QtWidgets.QLabel(
            "Genera un dataset sintético de prueba con los parámetros\n"
            "abajo y evalúa el modelo actual contra él. Útil sobre todo\n"
            "para modelos entrenados con datos sintéticos.\n\n"
            "Resultados (matriz, detalle por señal, métricas) se actualizan\n"
            "en las otras pestañas tras la evaluación."
        )
        info.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:9pt;")
        info.setWordWrap(True)
        ll.addWidget(info)

        # Generación
        grp = QtWidgets.QGroupBox("⚙  Generación del set de prueba")
        f = QtWidgets.QFormLayout(grp); f.setVerticalSpacing(8)
        self.reeval_n_samples = QtWidgets.QSpinBox()
        self.reeval_n_samples.setRange(50, 20000); self.reeval_n_samples.setValue(800)
        self.reeval_n_samples.setToolTip("Total de señales a generar (50% bypass / 50% normal).")
        self.reeval_duration = QtWidgets.QDoubleSpinBox()
        self.reeval_duration.setRange(1.0, 20.0); self.reeval_duration.setValue(5.0)
        self.reeval_duration.setSuffix(" s")
        self.reeval_duration.setToolTip("Duración de cada señal generada.")
        self.reeval_fs = QtWidgets.QSpinBox()
        self.reeval_fs.setRange(200, 5000); self.reeval_fs.setValue(2000)
        self.reeval_fs.setSuffix(" Hz")
        self.reeval_fs.setToolTip("Frecuencia de muestreo de cada señal generada.")
        self.reeval_seed = QtWidgets.QSpinBox()
        self.reeval_seed.setRange(0, 999999); self.reeval_seed.setValue(123)
        self.reeval_seed.setToolTip(
            "Semilla del generador. Cambiándola obtienes un set distinto\n"
            "con los mismos rangos."
        )
        f.addRow("Nº de muestras:", self.reeval_n_samples)
        f.addRow("Duración:",       self.reeval_duration)
        f.addRow("Muestreo (fs):",  self.reeval_fs)
        f.addRow("Semilla:",        self.reeval_seed)
        ll.addWidget(grp)

        # Rangos físicos
        self.reeval_range_panel = PhysicalRangePanel(
            "🎚️  Rangos físicos del transiente"
        )
        ll.addWidget(self.reeval_range_panel)

        # Botón ejecutar
        self.btn_run_reeval_syn = QtWidgets.QPushButton("🚀  Ejecutar re-evaluación sintética")
        self.btn_run_reeval_syn.setObjectName("primaryButton")
        self.btn_run_reeval_syn.setMinimumHeight(36)
        self.btn_run_reeval_syn.clicked.connect(self.on_reevaluate_synthetic)
        ll.addWidget(self.btn_run_reeval_syn)
        ll.addStretch()

        # — Lado derecho: log de la última ejecución —
        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)
        grp_log = QtWidgets.QGroupBox("📋  Resultado de la última re-evaluación")
        gl = QtWidgets.QVBoxLayout(grp_log)
        self.reeval_syn_log = QtWidgets.QPlainTextEdit()
        self.reeval_syn_log.setReadOnly(True)
        self.reeval_syn_log.setPlainText(
            "Aún no se ha ejecutado ninguna re-evaluación sintética.\n\n"
            "Configura los parámetros a la izquierda y pulsa\n"
            "«🚀 Ejecutar re-evaluación sintética»."
        )
        gl.addWidget(self.reeval_syn_log)
        rl.addWidget(grp_log)

        sp = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        sp.addWidget(scroll); sp.addWidget(right)
        sp.setSizes([420, 700])
        outer.addWidget(sp)
        return page

    def _build_reeval_experimental_tab(self) -> QtWidgets.QWidget:
        """Sub-pestaña con controles para re-evaluación experimental."""
        page = QtWidgets.QWidget()
        outer = QtWidgets.QHBoxLayout(page)
        outer.setContentsMargins(10, 10, 10, 10); outer.setSpacing(10)

        left = QtWidgets.QWidget()
        ll = QtWidgets.QVBoxLayout(left); ll.setContentsMargins(0, 0, 0, 0)

        info = QtWidgets.QLabel(
            "Re-evalúa el modelo actual contra señales reales etiquetadas\n"
            "por clase. Se calcula matriz de confusión, detalle por señal\n"
            "(con confianza) y se actualizan las otras pestañas con los\n"
            "resultados.\n\nFuente de datos:"
        )
        info.setStyleSheet(f"color:{COLOR_TEXT_DIM}; font-size:9pt;")
        info.setWordWrap(True)
        ll.addWidget(info)

        # Estado de los datos cargados en el entrenador real
        grp_status = QtWidgets.QGroupBox("📊  Datos disponibles")
        gs = QtWidgets.QVBoxLayout(grp_status)
        self.reeval_exp_status = QtWidgets.QLabel("…")
        self.reeval_exp_status.setStyleSheet("padding: 6px;")
        self.reeval_exp_status.setWordWrap(True)
        gs.addWidget(self.reeval_exp_status)
        ll.addWidget(grp_status)

        # Botones: usar datos del entrenador / cargar CSVs nuevos
        grp_actions = QtWidgets.QGroupBox("⚡  Acción")
        ga = QtWidgets.QVBoxLayout(grp_actions); ga.setSpacing(8)

        self.btn_reeval_use_existing = QtWidgets.QPushButton(
            "🔁  Usar datos del Entrenador real"
        )
        self.btn_reeval_use_existing.setToolTip(
            "Reutiliza las señales que ya están cargadas en la pestaña\n"
            "«Entrenamiento real». Requiere al menos una señal por clase."
        )
        self.btn_reeval_use_existing.clicked.connect(
            lambda: self._run_experimental_reeval(use_existing=True)
        )
        ga.addWidget(self.btn_reeval_use_existing)

        self.btn_reeval_load_new = QtWidgets.QPushButton(
            "📂  Cargar CSVs nuevos…"
        )
        self.btn_reeval_load_new.setToolTip(
            "Abre dos diálogos: primero pides los CSVs de clase Normal,\n"
            "luego los CSVs de clase Bypass. La re-evaluación se ejecuta\n"
            "con esos archivos sin tocar los del entrenador."
        )
        self.btn_reeval_load_new.clicked.connect(
            lambda: self._run_experimental_reeval(use_existing=False)
        )
        ga.addWidget(self.btn_reeval_load_new)
        ll.addWidget(grp_actions)
        ll.addStretch()

        right = QtWidgets.QWidget()
        rl = QtWidgets.QVBoxLayout(right); rl.setContentsMargins(0, 0, 0, 0)
        grp_log = QtWidgets.QGroupBox("📋  Resultado de la última re-evaluación")
        gl = QtWidgets.QVBoxLayout(grp_log)
        self.reeval_exp_log = QtWidgets.QPlainTextEdit()
        self.reeval_exp_log.setReadOnly(True)
        self.reeval_exp_log.setPlainText(
            "Aún no se ha ejecutado ninguna re-evaluación experimental.\n\n"
            "Selecciona una de las opciones a la izquierda."
        )
        gl.addWidget(self.reeval_exp_log)
        rl.addWidget(grp_log)

        sp = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        sp.addWidget(left); sp.addWidget(right)
        sp.setSizes([380, 700])
        outer.addWidget(sp)
        return page

    # ------------------------------------------------------------------
    def refresh(self):
        """Se llama cuando cambia el modelo en el main window."""
        mw = self.mw
        tr = mw.last_training
        avail = mw.available_model_keys()
        has_any = bool(avail)

        # Info banner
        if not has_any:
            self.info_label.setText(
                '<span style="color:{0};">Sin modelo cargado</span> — '
                'entrena en la pestaña «Entrenamiento…» o carga un .joblib.'
                .format(COLOR_TEXT_DIM)
            )
            self.report_text.setPlainText(
                "Aún no hay modelo disponible.\n\n"
                "Opciones:\n"
                "  1) Pestaña «Entrenamiento sintético» → botón entrenar.\n"
                "  2) Pestaña «Entrenamiento real» → carga CSVs y entrena.\n"
                "  3) Botón «📥 Cargar modelo» en la cabecera (.joblib).\n\n"
                "Tras cargar un modelo, podrás re-evaluarlo con:\n"
                "  • 🧪 Re-eval sintético (para modelos entrenados con sintético).\n"
                "  • 📂 Re-eval experimental (para modelos entrenados con datos reales)."
            )
            self.c_feat.show_empty()
            self.c_cm.reset_figure(); self.c_cm.show_empty()
            self.btn_run_reeval_syn.setEnabled(False)
            self.btn_reeval_use_existing.setEnabled(False)
            self.btn_reeval_load_new.setEnabled(False)
            self._update_reeval_exp_status()
            return

        self.btn_run_reeval_syn.setEnabled(True)
        self.btn_reeval_use_existing.setEnabled(True)
        self.btn_reeval_load_new.setEnabled(True)
        self._update_reeval_exp_status()

        # Fuente
        src_map = {
            "synthetic":   ("🧪", COLOR_ACCENT,  "Dataset sintético"),
            "real":        ("📂", COLOR_SUCCESS, "Dataset real"),
            "loaded":      ("📥", COLOR_WARNING, "Cargado desde .joblib"),
            "reeval":      ("🔄", COLOR_CYAN,    "Re-evaluado (sintético)"),
            "reeval_real": ("🔄", COLOR_SUCCESS, "Re-evaluado (experimental)"),
        }
        icon, color, label = src_map.get(
            tr.source if tr else "loaded", ("📥", COLOR_WARNING, "Cargado")
        )
        parts = []
        for k in avail:
            short = MODEL_SHORT_NAMES[k]
            acc_v = (tr.metrics.get(k) if tr else None)
            acc = f"{acc_v:.3f}" if acc_v is not None else "—"
            parts.append(f"<b>{short}</b> acc={acc}")
        samp = f" · n={tr.n_samples}" if tr and tr.n_samples else ""
        self.info_label.setText(
            f'<span style="color:{color};font-weight:700;">{icon} {label}</span>'
            f'{samp} &nbsp;|&nbsp; ' + " · ".join(parts)
        )

        self._render_report(tr, avail)
        self._render_feature_importance(tr, avail)
        self._render_confusion(tr, avail)
        self._render_per_sample(tr, avail)

    # ------------------------------------------------------------------
    # Detalle por muestra
    # ------------------------------------------------------------------
    def _render_per_sample(self, tr: Optional[TrainingResult],
                            avail: List[str]):
        """Llena la tabla detallada con una fila por señal evaluada.

        Las columnas son dinámicas: 3 fijas (#, Señal, Real) + 3 por
        cada modelo entrenado (pred, conf%, ✓), + 3 más para ensemble
        si hay 2+ modelos y hay datos de ensemble en per_sample.
        """
        # Estado de "no hay datos" o "no hay detalle"
        if tr is None or not tr.per_sample:
            self.tbl_per_sample.setSortingEnabled(False)
            self.tbl_per_sample.setRowCount(0)
            self.tbl_per_sample.setColumnCount(0)
            self.tbl_per_sample.setSortingEnabled(True)
            if tr is None:
                msg = "Sin datos. Carga un modelo y/o re-evalúa."
            else:
                msg = ("El modelo no tiene detalle por muestra disponible.\n"
                       "Pulsa «🔄 Re-eval sintético» o «📂 Re-eval experimental»\n"
                       "para generar predicciones individuales con su confianza.")
            self.lbl_per_sample_summary.setText(msg)
            return

        # Detectar qué keys de modelo aparecen realmente en per_sample.
        # Esto puede ser un subconjunto de `avail` (p. ej. si los datos
        # vienen de un .joblib viejo donde solo se guardó RF).
        keys_in_data: List[str] = []
        for k in MODEL_KEYS:
            if any(r.get(f"pred_{k}") is not None for r in tr.per_sample):
                keys_in_data.append(k)
        # Ensemble si tenemos datos de ensemble Y al menos 2 modelos
        has_ens_data = (
            len(keys_in_data) >= 2
            and any(r.get(f"pred_ensemble") is not None for r in tr.per_sample)
        )

        # Guardar el orden de columnas para que el filtro y el export lo usen
        self._per_sample_data = list(tr.per_sample)
        self._per_sample_keys = list(keys_in_data)
        self._per_sample_has_ensemble = has_ens_data
        self._apply_per_sample_filter()

    def _row_is_error_for(self, row: Dict[str, Any], key: str) -> bool:
        """Devuelve True si la predicción del modelo `key` falló."""
        true = row.get("true")
        pred = row.get(f"pred_{key}")
        return pred is not None and pred != true

    def _row_passes_filter(self, row: Dict[str, Any], mode: str) -> bool:
        keys = getattr(self, "_per_sample_keys", []) or []
        true = row.get("true")
        if mode == "Todas":
            return True
        if mode == "Solo aciertos (todos)":
            # Ningún modelo debe haber fallado; al menos uno debe haber predicho
            preds_present = any(row.get(f"pred_{k}") is not None for k in keys)
            return preds_present and not any(
                self._row_is_error_for(row, k) for k in keys
            )
        if mode == "Solo errores (cualquier modelo)":
            return any(self._row_is_error_for(row, k) for k in keys)
        if mode == "Solo clase Normal":
            return true == 0
        if mode == "Solo clase Bypass":
            return true == 1
        # Modos dinámicos por modelo: "Solo errores RF", etc.
        for k in keys:
            short = MODEL_SHORT_NAMES.get(k, k.upper())
            if mode == f"Solo errores {short}":
                return self._row_is_error_for(row, k)
        # Filtro de ensemble si está activo
        if getattr(self, "_per_sample_has_ensemble", False):
            if mode == "Solo errores ENS":
                return self._row_is_error_for(row, "ensemble")
        return True

    def _apply_per_sample_filter(self):
        """Aplica el filtro actual y reconstruye la tabla."""
        data = getattr(self, "_per_sample_data", None)
        if not data:
            return
        keys = getattr(self, "_per_sample_keys", []) or []
        has_ens = getattr(self, "_per_sample_has_ensemble", False)

        # Reconfigurar el combo de filtros si el conjunto de modelos
        # cambió desde la última vez (caso típico: cargar otro .joblib).
        self._reconfigure_filter_combo(keys, has_ens)

        mode = self.cm_filter_combo.currentText()
        rows = [r for r in data if self._row_passes_filter(r, mode)]

        # Resumen
        n_total  = len(data)
        n_normal = sum(1 for r in data if r.get("true") == 0)
        n_bypass = sum(1 for r in data if r.get("true") == 1)
        err_counts = {
            k: sum(1 for r in data if self._row_is_error_for(r, k))
            for k in keys
        }
        if has_ens:
            err_counts["ensemble"] = sum(
                1 for r in data if self._row_is_error_for(r, "ensemble")
            )
        err_summary = " · ".join(
            f"Err {MODEL_SHORT_NAMES.get(k, k.upper())}: {v}"
            for k, v in err_counts.items()
        )
        self.lbl_per_sample_summary.setText(
            f"Total: {n_total}  ·  Normal: {n_normal}  ·  Bypass: {n_bypass}"
            + (f"  ·  {err_summary}" if err_summary else "")
            + f"  ·  Mostrando: {len(rows)}"
        )

        # Construir las columnas dinámicamente
        col_keys = list(keys)
        if has_ens:
            col_keys.append("ensemble")

        headers = ["#", "Señal", "Real"]
        for k in col_keys:
            short = MODEL_SHORT_NAMES.get(k, k.upper())
            headers += [f"{short} pred", f"{short} conf %", f"{short} ✓"]

        labels = ["Normal", "Bypass"]
        bg_err   = QtGui.QColor("#3a2030")
        col_err  = QtGui.QColor(COLOR_DANGER)
        col_ok   = QtGui.QColor(COLOR_SUCCESS)
        col_text = QtGui.QColor(COLOR_TEXT)
        col_dim  = QtGui.QColor(COLOR_TEXT_DIM)

        def make_item(text: str, sort_value=None,
                       color: Optional[QtGui.QColor] = None,
                       bold: bool = False,
                       align=QtCore.Qt.AlignCenter):
            if sort_value is not None:
                it = _NumericTableItem(text, float(sort_value))
            else:
                it = QtWidgets.QTableWidgetItem(text)
            it.setTextAlignment(align)
            if color is not None:
                it.setForeground(QtGui.QBrush(color))
            if bold:
                f = it.font(); f.setBold(True); it.setFont(f)
            return it

        # Reconfigurar columnas y resize modes
        self.tbl_per_sample.setSortingEnabled(False)
        self.tbl_per_sample.setColumnCount(len(headers))
        self.tbl_per_sample.setHorizontalHeaderLabels(headers)
        ph = self.tbl_per_sample.horizontalHeader()
        ph.setSectionResizeMode(0, QtWidgets.QHeaderView.ResizeToContents)
        ph.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        for c in range(2, len(headers)):
            ph.setSectionResizeMode(c, QtWidgets.QHeaderView.ResizeToContents)

        self.tbl_per_sample.setRowCount(len(rows))

        for i, r in enumerate(rows):
            true = r.get("true")
            # Columnas 0, 1, 2
            self.tbl_per_sample.setItem(i, 0, make_item(str(i + 1), sort_value=i + 1))
            self.tbl_per_sample.setItem(i, 1, make_item(
                str(r.get("name", "?")),
                align=QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter
            ))
            self.tbl_per_sample.setItem(i, 2, make_item(
                labels[true] if true in (0, 1) else "?",
                color=col_text, bold=True
            ))

            row_has_error = False
            for c_idx, key in enumerate(col_keys):
                base_col = 3 + 3 * c_idx
                pred = r.get(f"pred_{key}")
                prob = r.get(f"prob_{key}_bypass")
                err  = pred is not None and pred != true
                if err:
                    row_has_error = True
                if pred is None:
                    self.tbl_per_sample.setItem(i, base_col,     make_item("—", color=col_dim))
                    self.tbl_per_sample.setItem(i, base_col + 1, make_item("—", color=col_dim))
                    self.tbl_per_sample.setItem(i, base_col + 2, make_item("—", color=col_dim))
                else:
                    self.tbl_per_sample.setItem(i, base_col, make_item(
                        labels[pred], color=col_err if err else col_text, bold=err
                    ))
                    if prob is not None:
                        conf = prob if pred == 1 else (1.0 - prob)
                        self.tbl_per_sample.setItem(i, base_col + 1, make_item(
                            f"{conf*100:.1f}%", sort_value=conf * 100,
                            color=col_err if err else col_text
                        ))
                    else:
                        self.tbl_per_sample.setItem(i, base_col + 1,
                                                    make_item("—", color=col_dim))
                    self.tbl_per_sample.setItem(i, base_col + 2, make_item(
                        "✗" if err else "✓",
                        color=col_err if err else col_ok, bold=True
                    ))

            if row_has_error:
                for c in range(self.tbl_per_sample.columnCount()):
                    it = self.tbl_per_sample.item(i, c)
                    if it is not None:
                        it.setBackground(QtGui.QBrush(bg_err))

        self.tbl_per_sample.setSortingEnabled(True)

    def _reconfigure_filter_combo(self, keys: List[str], has_ens: bool):
        """
        Repuebla el combo de filtros con opciones específicas para los
        modelos disponibles ("Solo errores RF", "Solo errores XGB", …)
        sin perder la selección actual cuando es posible.
        """
        new_options = ["Todas", "Solo errores (cualquier modelo)",
                       "Solo aciertos (todos)"]
        for k in keys:
            new_options.append(f"Solo errores {MODEL_SHORT_NAMES.get(k, k.upper())}")
        if has_ens:
            new_options.append("Solo errores ENS")
        new_options += ["Solo clase Normal", "Solo clase Bypass"]

        # Si el combo ya tiene exactamente esas opciones, no tocamos nada.
        current_items = [self.cm_filter_combo.itemText(i)
                          for i in range(self.cm_filter_combo.count())]
        if current_items == new_options:
            return

        # Repoblar conservando selección si sigue presente
        prev = self.cm_filter_combo.currentText()
        self.cm_filter_combo.blockSignals(True)
        self.cm_filter_combo.clear()
        self.cm_filter_combo.addItems(new_options)
        if prev in new_options:
            self.cm_filter_combo.setCurrentText(prev)
        self.cm_filter_combo.blockSignals(False)

    def _on_export_per_sample(self):
        """Exporta la tabla detallada (con filtro aplicado) a CSV.

        Las columnas exportadas son dinámicas según los modelos
        presentes en los datos (igual que la tabla en pantalla).
        """
        data = getattr(self, "_per_sample_data", [])
        if not data:
            QtWidgets.QMessageBox.information(
                self, "Exportar", "No hay detalle por muestra para exportar."
            )
            return
        mode = self.cm_filter_combo.currentText()
        rows = [r for r in data if self._row_passes_filter(r, mode)]
        if not rows:
            QtWidgets.QMessageBox.information(
                self, "Exportar",
                "El filtro actual no contiene ninguna fila."
            )
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Exportar detalle por muestra", "detalle_predicciones.csv",
            "CSV (*.csv);;Todos (*)"
        )
        if not path:
            return

        # Reusar las claves detectadas durante el render
        keys = list(getattr(self, "_per_sample_keys", []) or [])
        if getattr(self, "_per_sample_has_ensemble", False):
            keys.append("ensemble")

        try:
            labels = ["Normal", "Bypass"]
            with open(path, "w", encoding="utf-8") as fh:
                # Cabecera dinámica
                header_parts = ["indice", "nombre", "real"]
                for k in keys:
                    short = MODEL_SHORT_NAMES.get(k, k.upper()).lower()
                    header_parts += [f"{short}_pred",
                                      f"{short}_conf_pct",
                                      f"{short}_acierto"]
                fh.write(",".join(header_parts) + "\n")

                # Filas
                for i, r in enumerate(rows, 1):
                    true = r.get("true")
                    line_parts = [
                        str(i),
                        f'"{r.get("name", "")}"',
                        labels[true] if true in (0, 1) else "",
                    ]
                    for k in keys:
                        pred = r.get(f"pred_{k}")
                        prob = r.get(f"prob_{k}_bypass")
                        if pred in (0, 1):
                            pred_str = labels[pred]
                            if prob is not None:
                                conf = (prob if pred == 1 else 1.0 - prob) * 100
                                conf_str = f"{conf:.2f}"
                            else:
                                conf_str = ""
                            ok = "1" if pred == true else "0"
                        else:
                            pred_str, conf_str, ok = "", "", ""
                        line_parts += [pred_str, conf_str, ok]
                    fh.write(",".join(line_parts) + "\n")
            self.mw.status.showMessage(
                f"💾  Exportado: {os.path.basename(path)} ({len(rows)} filas)"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error al exportar", str(e))

    def _render_report(self, tr: Optional[TrainingResult],
                        avail: List[str]):
        if tr is None:
            self.report_text.setPlainText(
                "Modelo cargado sin métricas asociadas.\n\n"
                "Pulsa «🔄 Re-evaluar con sintético» para calcular\n"
                "accuracy, matriz de confusión e importancia sobre un\n"
                "set de prueba sintético."
            )
            return

        lines = []
        lines.append("═" * 48)
        lines.append(f"  ANÁLISIS DEL MODELO — fuente: {tr.source}")
        lines.append("═" * 48)
        if tr.n_samples:
            lines.append(f"Muestras de entrenamiento (total): {tr.n_samples}")
        lines.append(f"Features : {len(tr.feature_names)}")
        lines.append("")
        for k in avail:
            acc = tr.metrics.get(k)
            if acc is None:
                continue
            display = MODEL_DISPLAY_NAMES[k]
            lines.append(f"▶ {display}")
            lines.append(f"   Accuracy (hold-out) : {acc:.4f}")
            cv = tr.cvs.get(k)
            if cv is not None:
                lines.append(f"   CV 5-fold           : "
                             f"{cv.mean():.4f} ± {cv.std():.4f}")
            rep = tr.reports.get(k, "")
            if rep:
                lines.append("")
                lines.append(rep)
            lines.append("")
        # Si hay 2+ modelos, indicar disponibilidad de ensemble
        if len(avail) >= 2:
            lines.append("─" * 48)
            lines.append(
                f"💡 Ensemble (soft voting) disponible para predicciones."
            )
            lines.append(
                f"   Promedio probabilístico de: {', '.join(MODEL_SHORT_NAMES[k] for k in avail)}"
            )
        self.report_text.setPlainText("\n".join(lines))

    def _render_feature_importance(self, tr: Optional[TrainingResult],
                                    avail: List[str]):
        self.c_feat.clear_axes()
        ax = self.c_feat.axes[0, 0]
        if tr is None or tr.feature_importance is None:
            ax.text(0.5, 0.5,
                    "Importancia de características no disponible\n"
                    "(requiere RF, XGBoost o LightGBM entrenado)",
                    color=COLOR_TEXT_DIM, ha="center", va="center",
                    transform=ax.transAxes)
            self.c_feat.draw_idle()
            return
        imp   = tr.feature_importance
        names = tr.feature_names
        order = np.argsort(imp)
        top3  = set(np.argsort(imp)[-3:].tolist())
        bars = ax.barh(range(len(imp)), imp[order], color=COLOR_ACCENT, alpha=0.85)
        for i, idx in enumerate(order):
            if idx in top3:
                bars[i].set_color(COLOR_SUCCESS)
        ax.set_yticks(range(len(imp)))
        ax.set_yticklabels([names[i] for i in order], fontsize=8)
        # Identificar de qué modelo viene la importancia
        source_label = "Random Forest"
        if "rf" not in avail:
            if "xgb" in avail: source_label = "XGBoost"
            elif "lgbm" in avail: source_label = "LightGBM"
        ax.set_xlabel(f"Importancia ({source_label})")
        ax.set_title("Importancia de características", fontsize=10, pad=6)
        self.c_feat.draw_idle()

    def _render_confusion(self, tr: Optional[TrainingResult],
                           avail: List[str]):
        """
        Dibuja una matriz de confusión por cada modelo entrenado más
        una para el ensemble si hay 2+ modelos. La grid se reconstruye
        dinámicamente en función del número de modelos.
        """
        # Decidir qué matrices vamos a mostrar
        items: List[Tuple[str, str, Optional[np.ndarray]]] = []
        for k in avail:
            cm = tr.cms.get(k) if tr else None
            items.append((k, MODEL_DISPLAY_NAMES[k], cm))
        # ¿Hay ensemble cm calculado? (lo guardamos en cms["ensemble"])
        if tr is not None and len(avail) >= 2:
            cm_ens = tr.cms.get("ensemble")
            if cm_ens is not None:
                items.append(("ensemble", MODEL_DISPLAY_NAMES["ensemble"], cm_ens))

        n = max(1, len(items))
        # Reset completo de la figura para evitar superposición de
        # axes/colorbars residuales de redibujos previos.
        self.c_cm.fig.clear()
        self.c_cm.nrows, self.c_cm.ncols = 1, n
        self.c_cm.axes = self.c_cm.fig.subplots(1, n, squeeze=False)
        for ax in self.c_cm.axes[0]:
            _apply_ax_theme(ax)

        labels = ["Normal", "Bypass"]
        if not items:
            ax = self.c_cm.axes[0, 0]
            ax.text(0.5, 0.5, "Sin datos de matriz de confusión.",
                    color=COLOR_TEXT_DIM, ha="center", va="center",
                    transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
            self.c_cm.draw_idle()
            return

        for ax, (key, title, cm) in zip(self.c_cm.axes[0], items):
            short = MODEL_SHORT_NAMES.get(key, key.upper())
            full_title = f"{short} — {title}" if short != title else title
            if cm is None:
                ax.text(0.5, 0.5, f"Sin datos para {short}",
                        color=COLOR_TEXT_DIM, ha="center", va="center",
                        transform=ax.transAxes)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(full_title, fontsize=9, pad=6)
                continue
            ax.imshow(cm, cmap="Blues", aspect="auto")
            ax.set_title(full_title, fontsize=9, pad=6)
            ax.set_xticks([0, 1]); ax.set_xticklabels(labels, fontsize=8)
            ax.set_yticks([0, 1]); ax.set_yticklabels(labels, fontsize=8)
            ax.set_xlabel("Predicho", fontsize=8)
            ax.set_ylabel("Real", fontsize=8)
            vmax = cm.max() if cm.max() > 0 else 1
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    color = "white" if cm[i, j] > vmax / 2 else COLOR_TEXT
                    ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                            color=color, fontsize=12, fontweight="bold")
        self.c_cm.fig.tight_layout()
        self.c_cm.draw_idle()

    def on_reevaluate_synthetic(self):
        """
        Genera un dataset sintético usando los parámetros configurados
        en la sub-pestaña «Re-eval sintético» y evalúa el modelo actual
        contra él. Las métricas se actualizan en las otras pestañas.
        """
        mw = self.mw
        if not mw.has_any_model():
            return

        # Leer parámetros del panel
        n_samples   = self.reeval_n_samples.value()
        duration    = self.reeval_duration.value()
        fs          = self.reeval_fs.value()
        seed        = self.reeval_seed.value()
        param_ranges = self.reeval_range_panel.get_ranges()

        # Construir QProgressDialog. Importante: WA_DeleteOnClose para
        # que Qt limpie el objeto C++ aunque el usuario cierre la ventana
        # con la X. No mantenemos referencia tras este método (locales).
        progress = QtWidgets.QProgressDialog(
            "Generando dataset sintético de prueba…", "Cancelar", 0, 100, self
        )
        progress.setWindowTitle("Re-evaluación sintética")
        progress.setMinimumDuration(0)
        progress.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        progress.setValue(5)
        QtWidgets.QApplication.processEvents()

        try:
            X, y, names = generate_dataset(
                n_samples=n_samples, fs=fs, duration=duration, seed=seed,
                param_ranges=param_ranges,
                progress_cb=lambda p: (progress.setValue(5 + int(p * 0.7)),
                                       QtWidgets.QApplication.processEvents()),
            )

            # Validar features
            if mw.feature_names and mw.feature_names != names:
                missing = set(mw.feature_names) - set(names)
                if missing:
                    QtWidgets.QMessageBox.warning(
                        self, "Features incompatibles",
                        "El modelo espera features distintas a las generadas:\n"
                        f"{sorted(missing)}"
                    )
                    return
                idx = [names.index(n) for n in mw.feature_names]
                X = X[:, idx]
                names = mw.feature_names

            # Generar nombres simbólicos para el detalle por muestra
            sample_names = [f"synth_{i+1:04d}" for i in range(len(y))]

            tr = self._evaluate_model_against(
                X, y, names, sample_names=sample_names,
                source_tag="reeval", progress=progress
            )
            if tr is None:
                return

            mw.last_training = tr
            mw.model_updated.emit()
            mw.status.showMessage("🔄  Re-evaluación sintética completada.")

            # Log en la propia pestaña
            self._log_reeval_synthetic(tr, n_samples, fs, duration, seed,
                                        param_ranges)
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))
        finally:
            try:
                if progress is not None:
                    progress.close()
            except RuntimeError:
                # El objeto C++ ya fue eliminado: nada que hacer
                pass

    def _evaluate_model_against(self, X, y, feature_names: List[str],
                                  *, sample_names: List[str],
                                  source_tag: str,
                                  progress: Optional[QtWidgets.QProgressDialog] = None
                                  ) -> Optional["TrainingResult"]:
        """
        Pasa (X, y) por TODOS los modelos disponibles (rf/svm/xgb/lgbm)
        y construye un TrainingResult completo con métricas, matrices
        de confusión y per-sample.

        Si hay 2+ modelos, también calcula el ensemble (soft voting)
        y guarda sus métricas en la pseudo-clave 'ensemble'. El detalle
        por muestra contiene `pred_<key>` y `prob_<key>_bypass` para
        cada modelo individual y para el ensemble.

        Devuelve None si no hay ningún modelo cargado.
        """
        mw = self.mw
        avail = mw.available_model_keys()
        if not avail:
            return None

        Xs = mw.scaler.transform(X) if mw.scaler is not None else X

        tr = TrainingResult(
            models=dict(mw.models),
            scaler=mw.scaler, feature_names=feature_names,
            source=source_tag, n_samples=len(y),
        )

        # Predicciones por modelo
        per_model_pred: Dict[str, np.ndarray] = {}
        per_model_prob: Dict[str, Optional[np.ndarray]] = {}

        n_models = max(1, len(avail))
        for i, key in enumerate(avail):
            m = mw.models[key]
            try:
                yp = m.predict(Xs)
                pr = _prob_bypass(m, Xs)
            except Exception as e:
                # Si un modelo falla en predicción, lo saltamos pero
                # avisamos en el report
                tr.reports[key] = f"⚠️  {MODEL_DISPLAY_NAMES[key]} falló: {e}"
                continue
            per_model_pred[key] = yp
            per_model_prob[key] = pr
            tr.metrics[key] = accuracy_score(y, yp)
            tr.cms[key]     = confusion_matrix(y, yp, labels=[0, 1])
            tr.reports[key] = classification_report(
                y, yp, labels=[0, 1],
                target_names=["Normal", "Bypass"], zero_division=0
            )

            # Importancia: preferir RF, fallback a XGB/LGBM, descender por
            # CalibratedClassifierCV si fue calibrado
            if tr.feature_importance is None:
                inner = m
                if isinstance(m, CalibratedClassifierCV):
                    try:
                        inner = m.calibrated_classifiers_[0].estimator
                    except Exception:
                        inner = None
                if inner is not None and hasattr(inner, "feature_importances_"):
                    if key == "rf" or tr.feature_importance is None:
                        tr.feature_importance = np.asarray(
                            inner.feature_importances_, dtype=float
                        )

            if progress is not None:
                try:
                    progress.setValue(80 + int(15 * (i + 1) / n_models))
                except RuntimeError:
                    pass

        # Ensemble soft voting si hay 2+ modelos
        prob_ens = None
        pred_ens = None
        if len(per_model_prob) >= 2 and all(p is not None for p in per_model_prob.values()):
            prob_ens = np.mean(np.stack(list(per_model_prob.values()), axis=0), axis=0)
            pred_ens = (prob_ens >= 0.5).astype(int)
            tr.metrics[ENSEMBLE_KEY] = float(accuracy_score(y, pred_ens))
            tr.cms[ENSEMBLE_KEY] = confusion_matrix(y, pred_ens, labels=[0, 1])
            tr.reports[ENSEMBLE_KEY] = classification_report(
                y, pred_ens, labels=[0, 1],
                target_names=["Normal", "Bypass"], zero_division=0
            )

        # Detalle por muestra: una entrada por señal con todos los modelos
        tr.per_sample = []
        for idx in range(len(y)):
            row: Dict[str, Any] = {
                "name": sample_names[idx] if idx < len(sample_names) else f"#{idx+1}",
                "true": int(y[idx]),
            }
            for key in avail:
                yp = per_model_pred.get(key)
                pr = per_model_prob.get(key)
                row[f"pred_{key}"] = int(yp[idx]) if yp is not None else None
                row[f"prob_{key}_bypass"] = float(pr[idx]) if pr is not None else None
            if pred_ens is not None:
                row["pred_ensemble"] = int(pred_ens[idx])
                row["prob_ensemble_bypass"] = float(prob_ens[idx])
            tr.per_sample.append(row)

        if progress is not None:
            try:
                progress.setValue(100)
            except RuntimeError:
                pass
        return tr

    def _log_reeval_synthetic(self, tr: "TrainingResult", n_samples: int,
                                fs: int, duration: float, seed: int,
                                param_ranges: Dict[str, Tuple[float, float]]):
        """Escribe un resumen de la última re-evaluación sintética."""
        if not hasattr(self, "reeval_syn_log"):
            return
        lines = [
            "✅ Re-evaluación sintética completada.",
            "",
            f"Configuración usada:",
            f"  • Nº muestras : {n_samples}",
            f"  • Duración    : {duration:.2f} s",
            f"  • Muestreo fs : {fs} Hz",
            f"  • Semilla     : {seed}",
            "",
            "Rangos físicos:",
        ]
        for k, (lo, hi) in param_ranges.items():
            lines.append(f"  • {k:>10s}: [{lo:.4f}, {hi:.4f}]")
        lines.append("")
        # Métricas de TODOS los modelos disponibles
        avail = list(tr.metrics.keys())
        # Ordenar según MODEL_KEYS y al final ensemble
        ordered = [k for k in MODEL_KEYS if k in avail]
        if ENSEMBLE_KEY in avail:
            ordered.append(ENSEMBLE_KEY)
        for k in ordered:
            acc = tr.metrics.get(k)
            if acc is None:
                continue
            display = MODEL_DISPLAY_NAMES.get(k, k)
            lines.append(f"▶ {display:<25s} accuracy : {acc:.4f}")
        lines.append("")
        lines.append("Mira las pestañas «Métricas», «Matriz de confusión»")
        lines.append("y «Importancia» para los detalles.")
        self.reeval_syn_log.setPlainText("\n".join(lines))

    # ------------------------------------------------------------------
    # Re-evaluación con datos EXPERIMENTALES
    # ------------------------------------------------------------------
    def _update_reeval_exp_status(self):
        """Actualiza el QLabel de estado de la sub-pestaña experimental."""
        if not hasattr(self, "reeval_exp_status"):
            return
        mw = self.mw
        n_no  = len(mw.tab_real.data_no)  if hasattr(mw, "tab_real") else 0
        n_yes = len(mw.tab_real.data_yes) if hasattr(mw, "tab_real") else 0
        if n_no or n_yes:
            self.reeval_exp_status.setText(
                f'<span style="color:{COLOR_SUCCESS};">'
                f"En el «Entrenador real» hay actualmente:</span><br>"
                f"&nbsp;&nbsp;• <b>{n_no}</b> señal(es) Normal<br>"
                f"&nbsp;&nbsp;• <b>{n_yes}</b> señal(es) Bypass"
            )
        else:
            self.reeval_exp_status.setText(
                f'<span style="color:{COLOR_TEXT_DIM};">'
                f"No hay señales cargadas en «Entrenamiento real».<br>"
                f"Si quieres reutilizar datos, ve a esa pestaña primero<br>"
                f"y carga señales en ambas clases.</span>"
            )

    def on_reevaluate_experimental(self):
        """Punto de entrada legacy — abre el flujo del modal."""
        # Mantengo este método por si algo externo lo llama; ahora
        # delega en el método principal.
        self._run_experimental_reeval(use_existing=None)

    def _run_experimental_reeval(self, use_existing: Optional[bool]):
        """
        Ejecuta la re-evaluación experimental.
        - use_existing=True  → usa los datos del entrenador real.
        - use_existing=False → abre diálogos para cargar CSVs nuevos.
        - use_existing=None  → muestra el diálogo de elección clásico.
        """
        mw = self.mw
        if not mw.has_any_model():
            QtWidgets.QMessageBox.information(
                self, "Sin modelo", "Carga o entrena un modelo primero."
            )
            return
        if not mw.feature_names:
            QtWidgets.QMessageBox.warning(
                self, "Aviso",
                "El modelo cargado no tiene 'feature_names' definidos;\n"
                "no es posible construir el vector de features consistentemente."
            )
            return

        # Decidir fuente de datos
        if use_existing is None:
            data_no, data_yes = self._get_experimental_data()
            if data_no is None:
                return
        elif use_existing:
            data_no  = list(mw.tab_real.data_no)
            data_yes = list(mw.tab_real.data_yes)
            if not data_no or not data_yes:
                QtWidgets.QMessageBox.warning(
                    self, "Aviso",
                    "El entrenador real no tiene señales en ambas clases.\n"
                    "Carga señales primero o usa «Cargar CSVs nuevos…»."
                )
                return
        else:
            data_no, data_yes = self._load_experimental_csvs()
            if data_no is None:
                return

        if not data_no or not data_yes:
            QtWidgets.QMessageBox.warning(
                self, "Aviso",
                "Necesitas archivos válidos para AMBAS clases (Normal y Bypass)."
            )
            return

        self._run_experimental_evaluation(data_no, data_yes)

    def _get_experimental_data(self):
        """
        Devuelve (data_no, data_yes) o (None, None) si el usuario cancela.
        Ofrece reutilizar los datos de la pestaña «Entrenamiento real»
        si ya están cargados; en caso contrario, abre diálogos de archivo.
        """
        mw = self.mw
        existing_no  = list(mw.tab_real.data_no)
        existing_yes = list(mw.tab_real.data_yes)
        has_existing = bool(existing_no and existing_yes)

        if has_existing:
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Datos experimentales")
            box.setIcon(QtWidgets.QMessageBox.Question)
            box.setText(
                f"Hay datos experimentales cargados en «Entrenamiento real»:\n\n"
                f"   • {len(existing_no)} señal(es) Normal\n"
                f"   • {len(existing_yes)} señal(es) Bypass"
            )
            box.setInformativeText("¿Qué quieres usar para la re-evaluación?")
            btn_use    = box.addButton("Usar estos datos",   QtWidgets.QMessageBox.AcceptRole)
            btn_new    = box.addButton("Cargar CSVs nuevos", QtWidgets.QMessageBox.ActionRole)
            btn_cancel = box.addButton("Cancelar",           QtWidgets.QMessageBox.RejectRole)
            box.exec_()
            clicked = box.clickedButton()
            if clicked == btn_cancel:
                return None, None
            if clicked == btn_use:
                return existing_no, existing_yes
            # "Cargar nuevos" → sigue al diálogo de archivos abajo

        return self._load_experimental_csvs()

    def _load_experimental_csvs(self):
        """Abre dos diálogos para cargar CSVs Normal y Bypass."""
        files_no, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Señales de clase Normal — clase 0  (CSV / TXT / LOG)", "",
            SIGNAL_FILE_FILTER
        )
        if not files_no:
            return None, None
        files_yes, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Señales de clase Bypass — clase 1  (CSV / TXT / LOG)", "",
            SIGNAL_FILE_FILTER
        )
        if not files_yes:
            return None, None

        data_no, data_yes = [], []
        errors = []
        for f in files_no:
            try:
                t, p = load_csv_signal(f)
                data_no.append((f, t, p, infer_fs(t)))
            except Exception as e:
                errors.append(f"[Normal]  {os.path.basename(f)}: {e}")
        for f in files_yes:
            try:
                t, p = load_csv_signal(f)
                data_yes.append((f, t, p, infer_fs(t)))
            except Exception as e:
                errors.append(f"[Bypass]  {os.path.basename(f)}: {e}")

        if errors:
            QtWidgets.QMessageBox.warning(
                self, "Errores de lectura", "\n".join(errors)
            )
        return data_no, data_yes

    def _run_experimental_evaluation(self, data_no, data_yes):
        mw = self.mw
        n_total = len(data_no) + len(data_yes)

        # Aviso si el set es muy pequeño — las métricas serán poco confiables
        if n_total < 6:
            reply = QtWidgets.QMessageBox.question(
                self, "Set pequeño",
                f"Sólo se van a evaluar {n_total} señales en total.\n"
                f"Las métricas (accuracy, matriz de confusión) serán\n"
                f"poco estadísticamente robustas con tan pocas muestras.\n\n"
                f"¿Continuar de todas formas?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.Yes
            )
            if reply != QtWidgets.QMessageBox.Yes:
                return

        progress = QtWidgets.QProgressDialog(
            "Extrayendo features experimentales…", "Cancelar", 0, 100, self
        )
        progress.setWindowTitle("Re-evaluación experimental")
        progress.setMinimumDuration(0)
        progress.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        progress.setValue(2)
        QtWidgets.QApplication.processEvents()

        try:
            feature_names = mw.feature_names
            X_list, y_list, name_list, skipped = [], [], [], []

            done = 0
            for cls_value, group in [(0, data_no), (1, data_yes)]:
                for (path, t, p, fs) in group:
                    try:
                        feats = extract_features(t, p, fs)
                        X_list.append([feats.get(k, 0.0) for k in feature_names])
                        y_list.append(cls_value)
                        name_list.append(os.path.basename(path))
                    except Exception as e:
                        skipped.append(f"{os.path.basename(path)}: {e}")
                    done += 1
                    progress.setValue(2 + int(70 * done / max(1, n_total)))
                    QtWidgets.QApplication.processEvents()
                    if progress.wasCanceled():
                        return

            if skipped:
                QtWidgets.QMessageBox.warning(
                    self, "Señales omitidas",
                    "Algunas señales no se pudieron procesar:\n" + "\n".join(skipped)
                )

            if len(set(y_list)) < 2:
                QtWidgets.QMessageBox.warning(
                    self, "Clases insuficientes",
                    "Tras filtrar errores quedó sólo una clase representada;\n"
                    "no se puede construir una matriz de confusión 2x2."
                )
                return

            X = np.array(X_list, dtype=float)
            y = np.array(y_list, dtype=int)

            progress.setValue(75)

            tr = self._evaluate_model_against(
                X, y, feature_names,
                sample_names=name_list,
                source_tag="reeval_real",
                progress=progress,
            )
            if tr is None:
                return

            mw.last_training = tr
            mw.model_updated.emit()
            mw.status.showMessage(
                f"🔄  Re-evaluación experimental: {len(y)} señales "
                f"({int(np.sum(y == 0))} Normal · {int(np.sum(y == 1))} Bypass)."
            )

            # Log al panel de la sub-pestaña experimental
            if hasattr(self, "reeval_exp_log"):
                lines = [
                    "✅ Re-evaluación experimental completada.",
                    "",
                    f"Señales evaluadas: {len(y)}",
                    f"  • Normal : {int(np.sum(y == 0))}",
                    f"  • Bypass : {int(np.sum(y == 1))}",
                    "",
                ]
                avail = list(tr.metrics.keys())
                ordered = [k for k in MODEL_KEYS if k in avail]
                if ENSEMBLE_KEY in avail:
                    ordered.append(ENSEMBLE_KEY)
                for k in ordered:
                    acc = tr.metrics.get(k)
                    if acc is None:
                        continue
                    display = MODEL_DISPLAY_NAMES.get(k, k)
                    lines.append(f"▶ {display:<25s} accuracy : {acc:.4f}")
                lines.append("")
                lines.append("Detalle señal-a-señal en la pestaña «Matriz de confusión».")
                self.reeval_exp_log.setPlainText("\n".join(lines))

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self, "Error", f"{e}\n\n{traceback.format_exc()}"
            )
        finally:
            try:
                progress.close()
            except RuntimeError:
                pass


# ============================================================================
# 9. VENTANA PRINCIPAL
# ============================================================================

class HydroAnalyzerGUI(QtWidgets.QMainWindow):

    model_updated = QtCore.pyqtSignal()  # emitido cuando cambia el modelo

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(1400, 860)

        # Estado de modelo (compartido entre pestañas).
        # `models` es la fuente única de verdad y mapea clave → modelo.
        # Las propiedades `rf_model`/`svm_model` son shims retrocompatibles.
        self.models:        Dict[str, Any] = {}
        self.scaler:        Optional[StandardScaler] = None
        self.feature_names: Optional[List[str]] = None
        self.last_training: Optional[TrainingResult] = None

        # Hilo entrenamiento
        self._thread: Optional[QtCore.QThread] = None
        self._worker = None

        self._build_ui()
        self._apply_stylesheet()
        self.model_updated.connect(self._on_model_updated)
        # (v4.3) El fondo animado de símbolos ahora vive DENTRO del
        # header (AnimatedHeaderFrame, creado en _build_header) — el
        # antiguo overlay de ventana completa fue eliminado.

    # ── Properties de retrocompatibilidad ──────────────────────
    # El resto del código todavía hace `self.rf_model` y `self.svm_model`;
    # estas properties leen/escriben en self.models para no romper nada.
    @property
    def rf_model(self): return self.models.get("rf")
    @rf_model.setter
    def rf_model(self, val):
        if val is None: self.models.pop("rf", None)
        else: self.models["rf"] = val

    @property
    def svm_model(self): return self.models.get("svm")
    @svm_model.setter
    def svm_model(self, val):
        if val is None: self.models.pop("svm", None)
        else: self.models["svm"] = val

    @property
    def xgb_model(self): return self.models.get("xgb")
    @xgb_model.setter
    def xgb_model(self, val):
        if val is None: self.models.pop("xgb", None)
        else: self.models["xgb"] = val

    @property
    def lgbm_model(self): return self.models.get("lgbm")
    @lgbm_model.setter
    def lgbm_model(self, val):
        if val is None: self.models.pop("lgbm", None)
        else: self.models["lgbm"] = val

    def has_any_model(self) -> bool:
        return any(self.models.get(k) is not None for k in MODEL_KEYS)

    def available_model_keys(self) -> List[str]:
        return [k for k in MODEL_KEYS if self.models.get(k) is not None]


    # ---------------- UI ----------------
    def _build_ui(self):
        # (v4.4) El widget central es un QStackedWidget con dos páginas:
        #   0 → la aplicación normal (header + pestañas)
        #   1 → la página de Créditos a pantalla completa (sistema solar)
        # Nada de ventanas emergentes: los créditos viven DENTRO del
        # programa, como una sección más.
        self.main_stack = QtWidgets.QStackedWidget()
        self.setCentralWidget(self.main_stack)

        app_page = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(app_page)
        root.setContentsMargins(10, 10, 10, 10); root.setSpacing(8)

        root.addWidget(self._build_header())

        # Tabs principales
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setDocumentMode(True)

        self.tab_sim   = SimulatorTab(self)
        self.tab_syn   = SyntheticTrainerTab(self)
        self.tab_real  = RealTrainerTab(self)
        self.tab_anal  = ModelAnalysisTab(self)

        self.tabs.addTab(self.tab_sim,  "🔬  Simulador")
        self.tabs.addTab(self.tab_syn,  "🧪  Entrenamiento sintético")
        self.tabs.addTab(self.tab_real, "📂  Entrenamiento real")
        self.tabs.addTab(self.tab_anal, "📊  Análisis del modelo")
        root.addWidget(self.tabs, 1)

        # Página de créditos (sistema solar animado, render solo cuando
        # está visible gracias a showEvent/hideEvent)
        self.credits_page = CreditsPage()
        self.credits_page.back_requested.connect(self._leave_credits)

        # Página «Acerca de» (red neuronal 3D) — página 2
        self.about_page = AboutPage()
        self.about_page.back_requested.connect(self._leave_credits)

        self.main_stack.addWidget(app_page)           # índice 0
        self.main_stack.addWidget(self.credits_page)  # índice 1
        self.main_stack.addWidget(self.about_page)    # índice 2

        # ── v4.0: transiciones animadas entre pestañas ────────────────
        FX.attach_tab_fade(self.tabs)
        # Fades también en los visualizadores internos más usados
        FX.attach_tab_fade(getattr(self.tab_sim,  "viz_tabs", None))
        FX.attach_tab_fade(getattr(self.tab_real, "real_subtabs", None))
        FX.attach_tab_fade(getattr(self.tab_real, "cv_results_tabs", None))
        FX.attach_tab_fade(getattr(self.tab_anal, "viz_tabs", None))

        # ── v4.0: atajos de teclado ───────────────────────────────────
        for i in range(4):
            sc = QtWidgets.QShortcut(
                QtGui.QKeySequence(f"Ctrl+{i+1}"), self)
            sc.activated.connect(
                lambda idx=i: self.tabs.setCurrentIndex(idx))
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+O"), self)\
            .activated.connect(self.on_load_model)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+S"), self)\
            .activated.connect(self.on_save_model)
        QtWidgets.QShortcut(QtGui.QKeySequence("F1"), self)\
            .activated.connect(self.on_help)
        QtWidgets.QShortcut(QtGui.QKeySequence("Ctrl+G"), self)\
            .activated.connect(self.on_credits)

        # Status bar + progress
        self.status = QtWidgets.QStatusBar()
        self.setStatusBar(self.status)
        self.progress = QtWidgets.QProgressBar()
        self.progress.setFixedWidth(260); self.progress.setValue(0)
        self.status.addPermanentWidget(self.progress)
        self.status.showMessage(f"🟢 {APP_NAME} v{APP_VERSION} «Aurora» listo.")

    def _build_header(self):
        # (v4.3) El header es ahora un AnimatedHeaderFrame: pinta los
        # símbolos de fondo DETRÁS del título/badge/botones, recortados
        # a su rectángulo redondeado. setObjectName("header") ya lo hace
        # internamente, así que el stylesheet del gradiente aplica igual.
        hdr = AnimatedHeaderFrame()
        hdr.setMinimumHeight(62)
        lay = QtWidgets.QHBoxLayout(hdr)
        lay.setContentsMargins(18, 8, 18, 8)

        # Título + subtítulo — (v4.4) el título es CLICKEABLE: abre la
        # página de créditos a pantalla completa.
        col = QtWidgets.QVBoxLayout(); col.setSpacing(0)
        t = ClickableLabel("💧  HydroAnalyzer")
        t.setObjectName("headerTitle")
        t.setCursor(QtCore.Qt.PointingHandCursor)
        t.setToolTip("Click para ver los créditos  (Ctrl+G)")
        t.clicked.connect(self.on_credits)
        s = QtWidgets.QLabel("Análisis y clasificación ML de transientes hidráulicos")
        s.setObjectName("headerSubtitle")
        col.addWidget(t); col.addWidget(s)
        lay.addLayout(col); lay.addStretch()

        # Estado modelo
        self.model_badge = QtWidgets.QLabel("⚪  Sin modelo")
        self.model_badge.setObjectName("modelBadge")
        lay.addWidget(self.model_badge)
        lay.addSpacing(12)

        # Botones globales
        self.btn_load    = QtWidgets.QPushButton("📥  Cargar modelo")
        self.btn_save    = QtWidgets.QPushButton("💾  Guardar modelo")
        self.btn_credits = QtWidgets.QPushButton("❤  Créditos")
        self.btn_help    = QtWidgets.QPushButton("ℹ")
        self.btn_load.clicked.connect(self.on_load_model)
        self.btn_save.clicked.connect(self.on_save_model)
        self.btn_credits.clicked.connect(self.on_credits)
        self.btn_help.clicked.connect(self.on_help)
        self.btn_load.setObjectName("ghostButton")
        self.btn_save.setObjectName("ghostButton")
        self.btn_credits.setObjectName("ghostButton")
        self.btn_help.setObjectName("ghostButton")
        self.btn_help.setFixedWidth(36)
        self.btn_credits.setToolTip(
            f"Créditos y página web del autor (Ctrl+G)\n{AUTHOR_WEBSITE}"
        )
        for b in (self.btn_load, self.btn_save, self.btn_credits, self.btn_help):
            b.setCursor(QtCore.Qt.PointingHandCursor)
            lay.addWidget(b)

        # (v4.3) Se retiró el FX.glow del título: su drop-shadow con
        # blur 22 producía un "cajón" azulado de ~22px alrededor del
        # bloque de texto que tapaba los símbolos del fondo animado.
        # Con el background del header en movimiento, el texto nítido
        # luce mejor que con halo.
        return hdr

    def on_credits(self):
        """(v4.4) Muestra la página de Créditos a pantalla completa —
        ya no hay diálogo emergente. Se entra clickeando el título
        «HydroAnalyzer», el botón ❤ Créditos o Ctrl+G."""
        if self.main_stack.currentWidget() is self.credits_page:
            return
        self.main_stack.setCurrentWidget(self.credits_page)
        self.status.showMessage(
            "✨  Créditos — «← Volver» o Esc para regresar."
        )

    def _leave_credits(self):
        """Vuelve de la página de créditos a la aplicación normal."""
        self.main_stack.setCurrentIndex(0)
        FX.fade_in(self.main_stack.currentWidget(), FX.DURATION_FAST)
        self.status.showMessage(
            f"🟢 {APP_NAME} v{APP_VERSION} «Aurora» listo."
        )

    # ---------------- Training orchestration ----------------
    def start_training(self, worker):
        # Comprobar si hay un entrenamiento en curso. La comprobación es
        # delicada porque `self._thread` puede apuntar a un objeto C++ que
        # ya fue destruido por `deleteLater`; en ese caso `isRunning()`
        # lanza `RuntimeError: wrapped C/C++ object ... has been deleted`.
        # Tratamos ese caso como “no hay thread vivo”.
        if self._thread is not None:
            try:
                still_running = self._thread.isRunning()
            except RuntimeError:
                # El objeto C++ ya fue eliminado: limpiamos referencias
                # y seguimos como si no existiera.
                self._thread = None
                self._worker = None
                still_running = False
            if still_running:
                QtWidgets.QMessageBox.information(
                    self, "Info", "Ya hay un entrenamiento en curso."
                )
                return

        self._set_busy(True, "Iniciando entrenamiento…")

        self._thread = QtCore.QThread(self)
        self._worker = worker
        worker.moveToThread(self._thread)

        self._thread.started.connect(worker.run)
        worker.progress.connect(self._on_train_progress)
        worker.finished.connect(self._on_train_finished)
        worker.failed.connect(self._on_train_failed)
        worker.finished.connect(self._thread.quit)
        worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        # Cuando el thread termine, limpiar nuestras referencias Python
        # para evitar usar objetos C++ ya borrados en el próximo arranque.
        self._thread.finished.connect(self._on_thread_finished_cleanup)
        self._thread.start()

    @QtCore.pyqtSlot()
    def _on_thread_finished_cleanup(self):
        """Limpia las referencias al worker/thread tras finalizar.
        Sin esto, el siguiente `start_training` encontraría
        `self._thread` apuntando a un objeto C++ ya destruido por
        `deleteLater()` y `isRunning()` lanzaría RuntimeError."""
        self._thread = None
        self._worker = None

    @QtCore.pyqtSlot(int, str)
    def _on_train_progress(self, pct, msg):
        # v4.0: progreso suavizado — la barra se desliza al nuevo valor
        # en lugar de saltar, lo que hace los entrenamientos largos
        # visualmente más agradables.
        FX.animate_progress(self.progress, pct)
        self.status.showMessage(f"⚙  {msg}")

    @QtCore.pyqtSlot(object)
    def _on_train_finished(self, result: TrainingResult):
        # Volcar TODOS los modelos del resultado al estado del MW.
        # Empezamos limpiando: si un modelo no se entrenó esta vez,
        # lo eliminamos del MW para no mostrar info obsoleta.
        for k in MODEL_KEYS:
            self.models[k] = result.models.get(k)
            if self.models[k] is None:
                self.models.pop(k, None)
        self.scaler        = result.scaler
        self.feature_names = result.feature_names
        self.last_training = result
        self.model_updated.emit()
        self._set_busy(False, "✅  Entrenamiento finalizado.")
        # (v3.20) Sin redirección automática a la pestaña de análisis —
        # el usuario navega manualmente cuando quiera ver los resultados.

    @QtCore.pyqtSlot(str)
    def _on_train_failed(self, tb):
        self._set_busy(False, "❌  Error durante el entrenamiento.")
        QtWidgets.QMessageBox.critical(self, "Error de entrenamiento", tb)

    def _set_busy(self, busy, msg=""):
        for tab in (self.tab_syn, self.tab_real):
            tab.btn_train.setEnabled(not busy)
        if busy:
            self.progress.setValue(0)
        else:
            self.progress.setValue(0)
        if msg:
            self.status.showMessage(msg)

    # ---------------- Model save/load ----------------
    def on_save_model(self):
        if not self.has_any_model():
            QtWidgets.QMessageBox.warning(self, "Aviso", "No hay modelo para guardar.")
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Guardar modelo", "", "Joblib (*.joblib)"
        )
        if not path:
            return
        try:
            payload = {
                "format_version": MODEL_FORMAT_VERSION,
                "app_version": APP_VERSION,
                # Diccionario nuevo y canónico:
                "models": {k: self.models.get(k) for k in MODEL_KEYS
                            if self.models.get(k) is not None},
                # Aliases legacy para que un .joblib v3.14 pueda abrirse en
                # versiones anteriores del programa:
                "rf":  self.models.get("rf"),
                "svm": self.models.get("svm"),
                "scaler": self.scaler,
                "feature_names": self.feature_names,
                "metrics": self.last_training.to_dict() if self.last_training else None,
            }
            joblib.dump(payload, path)
            self.status.showMessage(f"💾  Modelo guardado: {os.path.basename(path)}")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    def on_load_model(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Cargar modelo", "", "Joblib (*.joblib)"
        )
        if not path:
            return
        try:
            data = joblib.load(path)
            if not isinstance(data, dict):
                raise ValueError("Formato de archivo no reconocido.")

            # Compat: aceptar tanto el dict canónico nuevo como los
            # aliases legacy 'rf'/'svm' del antiguo formato.
            self.models = {}
            models_dict = data.get("models")
            if isinstance(models_dict, dict):
                for k in MODEL_KEYS:
                    m = models_dict.get(k)
                    if m is not None:
                        self.models[k] = m
            # Aliases legacy
            for legacy_key in ("rf", "svm"):
                m = data.get(legacy_key)
                if m is not None and legacy_key not in self.models:
                    self.models[legacy_key] = m

            self.scaler        = data.get("scaler")
            self.feature_names = data.get("feature_names")

            # Reconstruir TrainingResult tolerante a ambos formatos
            metrics = data.get("metrics")
            if isinstance(metrics, dict):
                tr = TrainingResult(
                    models=dict(self.models),
                    scaler=self.scaler,
                    feature_names=self.feature_names or [],
                    feature_importance=metrics.get("feature_importance"),
                    source="loaded",
                    n_samples=metrics.get("n_samples", 0),
                    per_sample=metrics.get("per_sample", []) or [],
                )
                # Copiar métricas: preferir los dicts nuevos, hacer
                # fallback a los campos legacy si no están.
                for d_attr, key_legacy in [
                    ("metrics",  "accuracy_"),
                    ("cvs",      "cv_"),
                    ("cms",      "cm_"),
                    ("reports",  "report_"),
                ]:
                    new_val = metrics.get(d_attr)
                    if isinstance(new_val, dict):
                        getattr(tr, d_attr).update(new_val)
                    else:
                        # legacy: rellenar rf y svm desde claves planas
                        v_rf  = metrics.get(f"{key_legacy}rf")
                        v_svm = metrics.get(f"{key_legacy}svm")
                        if v_rf  is not None: getattr(tr, d_attr)["rf"]  = v_rf
                        if v_svm is not None: getattr(tr, d_attr)["svm"] = v_svm
                self.last_training = tr
            else:
                self.last_training = None

            self.model_updated.emit()
            self.status.showMessage(f"📥  Modelo cargado: {os.path.basename(path)}")

            # Aviso si no hay métricas
            if self.last_training is None:
                QtWidgets.QMessageBox.information(
                    self, "Modelo cargado",
                    "El modelo se cargó, pero el archivo no contiene métricas\n"
                    "(matriz de confusión, importancia, etc.).\n\n"
                    "En la pestaña «Análisis del modelo» puedes pulsar\n"
                    "«🔄 Re-evaluar con sintético» para calcularlas."
                )
            # (v3.20) Sin redirección automática — el usuario decide cuándo
            # ir a la pestaña de análisis.
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", str(e))

    @QtCore.pyqtSlot()
    def _on_model_updated(self):
        """Actualiza badge de cabecera y refresca análisis."""
        if not self.has_any_model():
            self.model_badge.setText("⚪  Sin modelo")
            self.model_badge.setStyleSheet(f"color:{COLOR_TEXT_DIM};")
        else:
            mods_short = [MODEL_SHORT_NAMES[k]
                          for k in self.available_model_keys()]
            txt = "🟢  " + "+".join(mods_short)
            if self.last_training:
                accs = []
                for k in self.available_model_keys():
                    acc = self.last_training.metrics.get(k)
                    if acc is not None:
                        accs.append(f"{MODEL_SHORT_NAMES[k]} {acc:.2f}")
                if accs:
                    txt += "  ·  " + " | ".join(accs)
            self.model_badge.setText(txt)
            self.model_badge.setStyleSheet(f"color:{COLOR_SUCCESS}; font-weight:700;")
            # v4.0: pequeño fade para que el cambio de modelo se note
            FX.fade_in(self.model_badge, FX.DURATION_MED)
        self.tab_anal.refresh()

    # ---------------- help ----------------
    def on_help(self):
        """(v4.6) Muestra la página «Acerca de» a pantalla completa con
        la red neuronal 3D — ya no es un diálogo emergente."""
        if self.main_stack.currentWidget() is self.about_page:
            return
        self.main_stack.setCurrentWidget(self.about_page)
        self.status.showMessage(
            "🧠  Acerca de HydroAnalyzer — «← Volver» o Esc para regresar."
        )

    # ---------------- estilo ----------------
    def _apply_stylesheet(self):
        self.setStyleSheet(f"""
        /* ════════ v4.0 «Aurora» ════════ */
        QMainWindow, QWidget {{
            background-color: {COLOR_BG}; color: {COLOR_TEXT};
            font-family: "Segoe UI", "Inter", "Roboto", sans-serif; font-size: 10pt;
        }}

        /* Header con degradado sutil */
        QFrame#header {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {COLOR_PANEL}, stop:0.55 #262b45, stop:1 {COLOR_PANEL_ALT});
            border: 1px solid {COLOR_BORDER}; border-radius: 12px;
        }}
        QFrame#infoFrame {{
            background-color: {COLOR_PANEL};
            border: 1px solid {COLOR_BORDER}; border-radius: 10px;
        }}
        QLabel#headerTitle    {{ font-size: 19pt; font-weight: 800; color: {COLOR_ACCENT};
                                 letter-spacing: 0.5px;
                                 background: transparent; }}
        QLabel#headerSubtitle {{ font-size: 9pt;  color: {COLOR_TEXT_DIM};
                                 background: transparent; }}
        QLabel#modelBadge     {{ font-size: 10pt; padding: 5px 14px;
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 {COLOR_PANEL_ALT}, stop:1 #181b2c);
            border: 1px solid {COLOR_BORDER}; border-radius: 15px;
        }}
        QLabel#infoLabel      {{ font-size: 11pt; }}
        QLabel#countsLabel    {{ color: {COLOR_ACCENT}; font-weight: 700;
            padding: 5px; border: 1px dashed {COLOR_BORDER}; border-radius: 6px;
            background-color: rgba(122, 162, 247, 0.06);
        }}

        /* Tarjetas */
        QGroupBox {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 {COLOR_PANEL}, stop:1 #20243a);
            border: 1px solid {COLOR_BORDER}; border-radius: 12px;
            margin-top: 15px; padding: 14px 11px 11px 11px;
            font-weight: 600; color: {COLOR_ACCENT};
        }}
        QGroupBox::title {{
            subcontrol-origin: margin; subcontrol-position: top left;
            left: 14px; padding: 0 8px; background-color: {COLOR_BG};
            border-radius: 4px;
        }}

        /* Botones */
        QPushButton {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 {COLOR_PANEL_ALT}, stop:1 #191c2e);
            color: {COLOR_TEXT};
            border: 1px solid {COLOR_BORDER}; border-radius: 8px;
            padding: 8px 14px;
        }}
        QPushButton:hover {{
            border-color: {COLOR_ACCENT}; color: {COLOR_ACCENT};
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #283052, stop:1 #1d2236);
        }}
        QPushButton:pressed {{ background-color: {COLOR_ACCENT}; color: #ffffff;
                               padding-top: 9px; padding-bottom: 7px; }}
        QPushButton:disabled{{ color: #565f89; border-color: #2a2f44;
                               background: #1b1e2e; }}
        QPushButton#primaryButton {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {COLOR_ACCENT}, stop:1 #5a7fd6);
            color: #0d0f1a; font-weight: 700;
            border: 1px solid {COLOR_ACCENT};
        }}
        QPushButton#primaryButton:hover {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {COLOR_CYAN}, stop:1 {COLOR_ACCENT});
            border-color: {COLOR_CYAN}; color: #0a0c14;
        }}
        QPushButton#primaryButton:pressed {{
            background: #5a7fd6;
        }}
        QPushButton#accentButton {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {COLOR_SUCCESS}, stop:1 #7fb356);
            color: #0d0f1a; font-weight: 700;
            border: 1px solid {COLOR_SUCCESS};
        }}
        QPushButton#accentButton:hover {{ background: #b4e08a; border-color: #b4e08a; }}
        QPushButton#ghostButton {{
            background: transparent; border: 1px solid {COLOR_BORDER};
            color: {COLOR_TEXT_DIM}; padding: 6px 13px;
        }}
        QPushButton#ghostButton:hover {{
            color: {COLOR_ACCENT}; border-color: {COLOR_ACCENT};
            background: rgba(122, 162, 247, 0.08);
        }}

        /* Inputs */
        QSpinBox, QDoubleSpinBox, QComboBox {{
            background-color: {COLOR_PANEL_ALT}; border: 1px solid {COLOR_BORDER};
            border-radius: 7px; padding: 5px 8px; color: {COLOR_TEXT};
            selection-background-color: {COLOR_ACCENT};
        }}
        QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
            border-color: #50598a;
        }}
        QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
            border: 1px solid {COLOR_ACCENT};
            background-color: #222741;
        }}
        QComboBox::drop-down {{ border: none; width: 22px; }}
        QComboBox QAbstractItemView {{
            background-color: {COLOR_PANEL}; color: {COLOR_TEXT};
            border: 1px solid {COLOR_BORDER}; border-radius: 6px;
            selection-background-color: {COLOR_ACCENT};
        }}

        QPlainTextEdit, QListWidget {{
            background-color: {COLOR_PANEL_ALT}; color: {COLOR_TEXT};
            border: 1px solid {COLOR_BORDER}; border-radius: 8px;
            font-family: "JetBrains Mono", "Consolas", monospace; font-size: 9pt;
            padding: 7px; selection-background-color: {COLOR_ACCENT};
        }}
        QListWidget::item {{ padding: 4px 6px; border-radius: 4px; }}
        QListWidget::item:hover {{ background-color: rgba(122,162,247,0.10); }}
        QListWidget::item:selected {{ background-color: {COLOR_ACCENT}; color: #ffffff; }}

        QCheckBox {{ spacing: 9px; color: {COLOR_TEXT}; }}
        QCheckBox::indicator {{
            width: 17px; height: 17px;
            border: 1px solid {COLOR_BORDER}; border-radius: 5px;
            background-color: {COLOR_PANEL_ALT};
        }}
        QCheckBox::indicator:hover {{ border-color: {COLOR_ACCENT}; }}
        QCheckBox::indicator:checked {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 {COLOR_ACCENT}, stop:1 {COLOR_CYAN});
            border-color: {COLOR_ACCENT};
        }}
        QRadioButton {{ spacing: 9px; color: {COLOR_TEXT}; }}
        QRadioButton::indicator {{
            width: 16px; height: 16px;
            border: 1px solid {COLOR_BORDER}; border-radius: 8px;
            background-color: {COLOR_PANEL_ALT};
        }}
        QRadioButton::indicator:checked {{
            background-color: {COLOR_ACCENT}; border-color: {COLOR_ACCENT};
        }}

        QStatusBar {{ background-color: {COLOR_PANEL}; color: {COLOR_TEXT};
                      border-top: 1px solid {COLOR_BORDER}; }}
        QProgressBar {{
            background-color: {COLOR_PANEL_ALT}; border: 1px solid {COLOR_BORDER};
            border-radius: 7px; text-align: center; color: {COLOR_TEXT};
            font-weight: 600; height: 16px;
        }}
        QProgressBar::chunk {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {COLOR_ACCENT}, stop:0.6 {COLOR_CYAN}, stop:1 {COLOR_MAGENTA});
            border-radius: 6px;
        }}

        /* Tabs */
        QTabWidget::pane {{ background: {COLOR_PANEL};
            border: 1px solid {COLOR_BORDER}; border-radius: 10px; top: -1px;
        }}
        QTabBar::tab {{
            background: transparent; color: {COLOR_TEXT_DIM};
            padding: 9px 20px; margin-right: 4px;
            border: 1px solid transparent; border-bottom: none;
            border-top-left-radius: 9px; border-top-right-radius: 9px;
        }}
        QTabBar::tab:selected {{
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 {COLOR_PANEL}, stop:1 #232842);
            color: {COLOR_ACCENT}; font-weight: 700;
            border: 1px solid {COLOR_BORDER}; border-bottom: none;
        }}
        QTabBar::tab:hover:!selected {{
            color: {COLOR_TEXT};
            background: rgba(122,162,247,0.07);
        }}

        /* Tablas y árboles */
        QTableWidget, QTreeWidget {{
            background-color: {COLOR_PANEL_ALT};
            alternate-background-color: #222740;
            color: {COLOR_TEXT};
            border: 1px solid {COLOR_BORDER}; border-radius: 8px;
            gridline-color: #2c3150;
            selection-background-color: rgba(122,162,247,0.30);
            selection-color: {COLOR_TEXT};
        }}
        QHeaderView::section {{
            background-color: {COLOR_PANEL};
            color: {COLOR_ACCENT}; font-weight: 700;
            border: none; border-bottom: 2px solid {COLOR_BORDER};
            padding: 6px 8px;
        }}
        QTreeWidget::branch {{ background: transparent; }}

        QSplitter::handle {{ background-color: {COLOR_BG}; width: 6px; }}
        QSplitter::handle:hover {{ background-color: {COLOR_ACCENT}; }}

        QToolBar {{
            background: {COLOR_PANEL_ALT}; border: 1px solid {COLOR_BORDER};
            border-radius: 8px; padding: 3px; spacing: 3px;
        }}
        QToolBar QToolButton {{ background: transparent; padding: 5px;
                                border-radius: 5px; }}
        QToolBar QToolButton:hover {{ background: rgba(122,162,247,0.14); }}

        /* Scrollbars finos */
        QScrollBar:vertical   {{ background: transparent; width: 9px; margin: 2px; }}
        QScrollBar::handle:vertical {{ background: {COLOR_BORDER};
            border-radius: 4px; min-height: 24px; }}
        QScrollBar::handle:vertical:hover {{ background: {COLOR_ACCENT}; }}
        QScrollBar:horizontal {{ background: transparent; height: 9px; margin: 2px; }}
        QScrollBar::handle:horizontal {{ background: {COLOR_BORDER};
            border-radius: 4px; min-width: 24px; }}
        QScrollBar::handle:horizontal:hover {{ background: {COLOR_ACCENT}; }}
        QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
        QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

        QScrollArea {{ background: transparent; border: none; }}

        QToolTip {{
            background-color: {COLOR_PANEL}; color: {COLOR_TEXT};
            border: 1px solid {COLOR_ACCENT}; border-radius: 6px;
            padding: 6px 8px;
        }}

        QMessageBox, QDialog {{ background-color: {COLOR_PANEL}; }}
        """)


# ============================================================================
# 10. ENTRY POINT
# ============================================================================

def main():
    app = QtWidgets.QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)

    pal = QtGui.QPalette()
    pal.setColor(QtGui.QPalette.Window,        QtGui.QColor(COLOR_BG))
    pal.setColor(QtGui.QPalette.WindowText,    QtGui.QColor(COLOR_TEXT))
    pal.setColor(QtGui.QPalette.Base,          QtGui.QColor(COLOR_PANEL_ALT))
    pal.setColor(QtGui.QPalette.AlternateBase, QtGui.QColor(COLOR_PANEL))
    pal.setColor(QtGui.QPalette.Text,          QtGui.QColor(COLOR_TEXT))
    pal.setColor(QtGui.QPalette.Button,        QtGui.QColor(COLOR_PANEL))
    pal.setColor(QtGui.QPalette.ButtonText,    QtGui.QColor(COLOR_TEXT))
    pal.setColor(QtGui.QPalette.Highlight,     QtGui.QColor(COLOR_ACCENT))
    pal.setColor(QtGui.QPalette.HighlightedText, QtGui.QColor("#ffffff"))
    pal.setColor(QtGui.QPalette.ToolTipBase,   QtGui.QColor(COLOR_PANEL))
    pal.setColor(QtGui.QPalette.ToolTipText,   QtGui.QColor(COLOR_TEXT))
    app.setPalette(pal)

    # ── v4.0: splash screen «Aurora» ──────────────────────────────────
    # Desactivable con HYDRO_NO_SPLASH=1 (útil para tests automatizados
    # y para usuarios que prefieran arranque instantáneo).
    use_splash = os.environ.get("HYDRO_NO_SPLASH", "0") != "1"
    splash = None
    if use_splash:
        splash = AuroraSplash()
        splash.show()
        app.processEvents()

    win = HydroAnalyzerGUI()

    def _reveal():
        win.show()
        FX.fade_in(win.centralWidget(), FX.DURATION_SLOW)
        if splash is not None:
            splash.fade_out_and_close()

    if use_splash:
        # Mantener el splash visible ~1.1 s (tiempo de marca + carga real)
        QtCore.QTimer.singleShot(1100, _reveal)
    else:
        _reveal()

    sys.exit(app.exec_())


if __name__ == "__main__":
    # (v4.5) OBLIGATORIO para ejecutables congelados en Windows
    # (PyInstaller / py2exe): cuando joblib/loky lanza procesos hijos,
    # estos RE-EJECUTAN el .exe completo. freeze_support() detecta esos
    # relanzamientos AL INICIO y los desvía a su rol de worker en lugar
    # de dejar que ejecuten main() (lo que abría una ventana nueva de
    # HydroAnalyzer por cada worker y bloqueaba el entrenamiento).
    # En ejecución normal (python hydroanalyzer.py) es un no-op inocuo.
    multiprocessing.freeze_support()
    main()
