"""
instalador.py  -  Instalador GUI de UNIFIEDVX.

Descarga las 2 partes desde GitHub Releases, las une en un solo .exe
en la carpeta que elija el usuario y verifica el SHA256.

Solo usa la libreria estandar de Python (tkinter + urllib), asi que se
convierte a .exe sin dependencias extra.

Para generar el .exe:
    pip install pyinstaller
    pyinstaller --onefile --windowed --name "Instalador UNIFIEDVX" instalador.py
"""

import hashlib
import os
import shutil
import threading
import urllib.request
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, font as tkfont

# ==================== CONFIGURACION (RELLENA ESTO) ====================
# URLs de descarga directa de cada parte. Tras crear los releases, la URL es:
#   https://github.com/<USUARIO>/<REPO>/releases/download/<TAG>/<NOMBRE_ASSET>
PART1_URL = "https://github.com/AndyRCC/HydraDetect-AI/releases/download/v4.6/UNIFIEDVX.part1"
PART2_URL = "https://github.com/AndyRCC/HydraDetect-AI/releases/download/v4.6/UNIFIEDVX.part2"

FINAL_NAME = "UNIFIEDVX.exe"            # nombre del exe final
EXPECTED_SHA256 = "02b4e4767315b558415f8ee5ef92b561ab418dae5bbc5cec176a7cbf0d117eae"  # lo imprime dividir.py
EXPECTED_TOTAL_SIZE = 2939203330       # bytes; lo imprime dividir.py (para la barra y el chequeo de disco)

CHUNK = 1024 * 1024                      # 1 MB por bloque de descarga
APP_TITLE = "Instalador UNIFIEDVX"
# =====================================================================

# ----------------------------- Paleta -------------------------------
BG          = "#0A1A24"   # fondo abismo
HEADER      = "#0C2029"   # banda de cabecera
SURFACE     = "#102A36"   # campos / superficies
TRACK       = "#0C1F29"   # riel de la barra
HAIRLINE    = "#1C3A47"   # bordes finos
TEXT        = "#EAF4F7"   # texto principal
MUTED       = "#6F94A3"   # texto secundario
A1          = "#34D399"   # acento parte 1 (esmeralda)
A1_HOVER    = "#54E0AC"
A2          = "#9B8CFA"   # acento parte 2 (violeta)
ERROR       = "#F87171"
BTN_FG      = "#06231A"   # texto sobre el boton esmeralda
GHOST_HOVER = "#163542"
DISABLED_BG = "#16313D"
DISABLED_FG = "#4E6B78"


def human(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.2f} {unit}"
        n /= 1024
    return f"{n:.2f} PB"


def pick_font(candidates, available):
    for name in candidates:
        if name in available:
            return name
    return candidates[-1]


def round_rect_points(x1, y1, x2, y2, r):
    """Puntos para un rectangulo redondeado con create_polygon(smooth=True)."""
    return [
        x1 + r, y1, x2 - r, y1, x2, y1, x2, y1 + r,
        x2, y2 - r, x2, y2, x2 - r, y2, x1 + r, y2,
        x1, y2, x1, y2 - r, x1, y1 + r, x1, y1,
    ]


class RoundedButton(tk.Canvas):
    """Boton plano con esquinas redondeadas dibujado sobre un canvas."""

    def __init__(self, parent, text, command, *, width, height, font,
                 fill, fg, hover, container_bg, border=None, radius=12):
        super().__init__(parent, width=width, height=height,
                         bg=container_bg, highlightthickness=0, bd=0)
        self.command = command
        self._bw, self._bh, self._br = width, height, radius
        self._fill, self._fg, self._hover, self._border = fill, fg, hover, border
        self._font, self._text = font, text
        self._enabled = True
        self._cur = fill
        self._draw()
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _draw(self):
        self.delete("all")
        pts = round_rect_points(1, 1, self._bw - 1, self._bh - 1, self._br)
        self._shape = self.create_polygon(
            pts, smooth=True, fill=self._cur,
            outline=self._border or self._cur,
        )
        self._label = self.create_text(
            self._bw / 2, self._bh / 2, text=self._text, fill=self._fg, font=self._font,
        )

    def _on_enter(self, _):
        if self._enabled:
            self._cur = self._hover
            self.itemconfig(self._shape, fill=self._cur)
            self.config(cursor="hand2")

    def _on_leave(self, _):
        if self._enabled:
            self._cur = self._fill
            self.itemconfig(self._shape, fill=self._cur)
            self.config(cursor="")

    def _on_click(self, _):
        if self._enabled and self.command:
            self.command()

    def set_enabled(self, enabled):
        self._enabled = enabled
        if enabled:
            self._cur = self._fill
            self.itemconfig(self._shape, fill=self._fill,
                            outline=self._border or self._fill)
            self.itemconfig(self._label, fill=self._fg)
        else:
            self._cur = DISABLED_BG
            self.itemconfig(self._shape, fill=DISABLED_BG, outline=DISABLED_BG)
            self.itemconfig(self._label, fill=DISABLED_FG)
            self.config(cursor="")

    def reconfigure(self, text=None, command=None):
        if text is not None:
            self._text = text
            self.itemconfig(self._label, text=text)
        if command is not None:
            self.command = command


class Installer:
    PW = 504          # ancho util / ancho del canvas de progreso
    NODE_MIN = 4.0

    def __init__(self, root):
        self.root = root
        self.dest = None
        self.dest_final = None
        self.written = 0
        self.cancelled = False
        self.running = False
        self._node_r = self.NODE_MIN
        self._pulse_phase = 0
        self._fonts()
        self._build_ui()
        self._draw_progress()
        self._center()

    # ----------------------------- Fuentes -----------------------------
    def _fonts(self):
        fams = set(tkfont.families())
        disp = pick_font(["Bahnschrift SemiBold", "Bahnschrift",
                          "Segoe UI Semibold", "Segoe UI"], fams)
        body = pick_font(["Segoe UI", "Tahoma"], fams)
        mono = pick_font(["Consolas", "Cascadia Mono", "Courier New"], fams)
        self.F_WORD   = (disp, 23)
        self.F_EYEBROW = (mono, 8)
        self.F_LABEL  = (mono, 8)
        self.F_BODY   = (body, 11)
        self.F_BTN    = (body, 11, "bold")
        self.F_SUB    = (mono, 9)
        self.F_STATUS = (mono, 10)
        self.F_META   = (mono, 8)

    # ----------------------------- UI -----------------------------
    def _build_ui(self):
        self.root.title(APP_TITLE)
        self.root.geometry("560x474")
        self.root.resizable(False, False)
        self.root.configure(bg=BG)

        self._build_header()
        tk.Frame(self.root, height=1, bg=HAIRLINE).pack(fill="x")
        self._build_body()
        self._build_footer()

    def _build_header(self):
        head = tk.Frame(self.root, bg=HEADER)
        head.pack(fill="x")

        logo = tk.Canvas(head, width=46, height=46, bg=HEADER,
                         highlightthickness=0, bd=0)
        logo.pack(side="left", padx=(22, 14), pady=20)
        # Dos cuñas que apuntan a un nodo central: dos partes que se fusionan.
        logo.create_polygon(4, 9, 4, 37, 20, 23, fill=A1, outline=A1)
        logo.create_polygon(42, 9, 42, 37, 26, 23, fill=A2, outline=A2)
        logo.create_oval(20, 19, 26, 25, fill=TEXT, outline=TEXT)

        txt = tk.Frame(head, bg=HEADER)
        txt.pack(side="left", anchor="w", pady=20)
        word = tk.Frame(txt, bg=HEADER)
        word.pack(anchor="w")
        tk.Label(word, text="UNIFIED", fg=TEXT, bg=HEADER, font=self.F_WORD).pack(side="left")
        tk.Label(word, text="VX", fg=A1, bg=HEADER, font=self.F_WORD).pack(side="left")
        tk.Label(txt, text="Instalador  ·  HydraDetect-AI", fg=MUTED, bg=HEADER,
                 font=self.F_SUB).pack(anchor="w", pady=(2, 0))

    def _eyebrow(self, parent, text):
        row = tk.Frame(parent, bg=BG)
        tk.Frame(row, width=3, height=12, bg=A1).pack(side="left")
        tk.Label(row, text=text, fg=MUTED, bg=BG, font=self.F_EYEBROW).pack(
            side="left", padx=(8, 0))
        return row

    def _build_body(self):
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=28, pady=(20, 0))

        # --- Carpeta de destino ---
        self._eyebrow(body, "CARPETA DE DESTINO").pack(anchor="w")
        row = tk.Frame(body, bg=BG)
        row.pack(fill="x", pady=(8, 0))

        self.browse_btn = RoundedButton(
            row, "Examinar", self.choose_folder, width=116, height=40,
            font=self.F_BTN, fill=SURFACE, fg=TEXT, hover=GHOST_HOVER,
            container_bg=BG, border=HAIRLINE, radius=10)
        self.browse_btn.pack(side="right", padx=(10, 0))

        wrap = tk.Frame(row, bg=SURFACE, highlightthickness=1,
                        highlightbackground=HAIRLINE, highlightcolor=A1)
        wrap.pack(side="left", fill="x", expand=True)
        self.path_var = tk.StringVar(value="Selecciona una carpeta de destino")
        entry = tk.Entry(wrap, textvariable=self.path_var, state="readonly",
                         relief="flat", bd=0, bg=SURFACE, readonlybackground=SURFACE,
                         fg=TEXT, disabledforeground=TEXT, font=self.F_BODY)
        entry.pack(fill="x", padx=12, ipady=9)

        # --- Progreso (firma: fusion de dos partes) ---
        self._eyebrow(body, "PROGRESO").pack(anchor="w", pady=(22, 0))
        self.canvas = tk.Canvas(body, width=self.PW, height=58, bg=BG,
                                highlightthickness=0, bd=0)
        self.canvas.pack(anchor="w", pady=(8, 0))
        self._draw_track()

        self.status_var = tk.StringVar(value="Listo para instalar.")
        self.status_lbl = tk.Label(body, textvariable=self.status_var, fg=MUTED,
                                   bg=BG, font=self.F_STATUS, anchor="w")
        self.status_lbl.pack(anchor="w", pady=(6, 0))

        # --- Accion ---
        self.install_btn = RoundedButton(
            body, "Instalar", self.start_install, width=156, height=44,
            font=self.F_BTN, fill=A1, fg=BTN_FG, hover=A1_HOVER,
            container_bg=BG, radius=12)
        self.install_btn.pack(anchor="e", pady=(18, 0))

    def _build_footer(self):
        tk.Frame(self.root, height=1, bg=HAIRLINE).pack(fill="x")
        foot = tk.Frame(self.root, bg=BG)
        foot.pack(fill="x")
        tk.Label(foot, text=f"≈ {human(EXPECTED_TOTAL_SIZE)}   ·   se verifica con SHA-256",
                 fg=MUTED, bg=BG, font=self.F_META).pack(anchor="w", padx=28, pady=11)

    # --------------------- Dibujo de la barra --------------------
    def _draw_track(self):
        c = self.canvas
        w, h, cy = self.PW, 14, 40
        top, bot, mid = cy - h / 2, cy + h / 2, w / 2
        c.create_polygon(round_rect_points(0, top, w, bot, h / 2),
                         smooth=True, fill=TRACK, outline=HAIRLINE)
        c.create_line(mid, top - 4, mid, bot + 4, fill=HAIRLINE)
        c.create_text(mid / 2, 12, text="PARTE 1", fill=A1, font=self.F_LABEL)
        c.create_text(mid + mid / 2, 12, text="PARTE 2", fill=A2, font=self.F_LABEL)

    def _pill(self, x1, x2, color):
        h, cy = 14, 40
        top, bot = cy - h / 2, cy + h / 2
        r = min(h / 2, (x2 - x1) / 2)
        self.canvas.create_polygon(round_rect_points(x1, top, x2, bot, r),
                                   smooth=True, fill=color, outline=color, tags="fill")

    def _draw_progress(self):
        c = self.canvas
        c.delete("fill")
        c.delete("node")
        w, cy, mid = self.PW, 40, self.PW / 2
        fx = max(0.0, min(w, self.written / EXPECTED_TOTAL_SIZE * w)) if EXPECTED_TOTAL_SIZE else 0
        p1end = min(fx, mid)
        if p1end > 2:
            self._pill(0, p1end, A1)
        if fx > mid + 1:
            self._pill(mid, fx, A2)
        if fx >= w - 0.5:   # fusionado
            c.create_oval(mid - 7, cy - 7, mid + 7, cy + 7,
                          fill=TEXT, outline=A1, width=2, tags="node")
        else:
            nr = self._node_r
            c.create_oval(mid - nr, cy - nr, mid + nr, cy + nr,
                          fill=HEADER, outline=MUTED, width=2, tags="node")

    def _pulse(self):
        if not self.running:
            return
        self._pulse_phase = (self._pulse_phase + 1) % 20
        p = self._pulse_phase
        self._node_r = self.NODE_MIN + 3.0 * (p if p <= 10 else 20 - p) / 10.0
        self._draw_progress()
        self.root.after(90, self._pulse)

    # --------------------- Acciones del usuario --------------------
    def choose_folder(self):
        folder = filedialog.askdirectory(title="Elige donde instalar")
        if folder:
            self.dest = folder
            self.path_var.set(folder)

    def start_install(self):
        if self.running:
            return
        if "USUARIO/REPO" in PART1_URL or EXPECTED_SHA256 == "PEGA_AQUI_EL_SHA256":
            messagebox.showerror(
                APP_TITLE,
                "El instalador no esta configurado.\n\n"
                "Edita PART1_URL, PART2_URL, EXPECTED_SHA256 y "
                "EXPECTED_TOTAL_SIZE en instalador.py antes de compilarlo.")
            return
        if not self.dest:
            messagebox.showwarning(APP_TITLE, "Primero elige la carpeta de instalacion.")
            return

        final_path = os.path.join(self.dest, FINAL_NAME)
        if os.path.exists(final_path):
            if not messagebox.askyesno(
                    APP_TITLE, f"Ya existe:\n{final_path}\n\n¿Sobreescribir?"):
                return

        try:
            free = shutil.disk_usage(self.dest).free
        except OSError:
            free = None
        if free is not None and free < EXPECTED_TOTAL_SIZE * 1.05:
            messagebox.showerror(
                APP_TITLE,
                f"Espacio insuficiente.\n\nSe necesitan ~{human(EXPECTED_TOTAL_SIZE)} "
                f"y solo hay {human(free)} libres en el disco destino.")
            return

        self.running = True
        self.cancelled = False
        self.written = 0
        self._set_buttons(False)
        self._status("Conectando…")
        self._pulse_phase = 0
        self._pulse()
        threading.Thread(target=self._worker, args=(final_path,), daemon=True).start()

    # ------------------------- Trabajo --------------------------
    def _worker(self, final_path):
        sha = hashlib.sha256()
        try:
            with open(final_path, "wb") as fh:
                self._ui(self._status, "Descargando parte 1 de 2…")
                self._download_into(PART1_URL, fh, sha)
                self._ui(self._status, "Descargando parte 2 de 2…")
                self._download_into(PART2_URL, fh, sha)

            self._ui(self._status, "Verificando integridad (SHA-256)…")
            if sha.hexdigest().lower() != EXPECTED_SHA256.lower():
                self._safe_remove(final_path)
                self._fail("La verificacion SHA-256 fallo: la descarga esta "
                           "corrupta o incompleta. Vuelve a intentarlo.")
                return

            self._ui(self._on_success, final_path)
        except Exception as e:
            self._safe_remove(final_path)
            self._fail(f"Error durante la instalacion:\n{e}")

    def _download_into(self, url, fh, sha):
        req = urllib.request.Request(url, headers={"User-Agent": "UNIFIEDVX-Installer"})
        with urllib.request.urlopen(req) as resp:
            while True:
                chunk = resp.read(CHUNK)
                if not chunk:
                    break
                if self.cancelled:
                    raise RuntimeError("Instalacion cancelada.")
                fh.write(chunk)
                sha.update(chunk)
                self.written += len(chunk)
                self._ui(self._update_progress)

    # --------------------- Helpers de UI/hilo --------------------
    def _ui(self, func, *args):
        """Ejecuta una actualizacion de UI en el hilo principal de tkinter."""
        self.root.after(0, lambda: func(*args))

    def _update_progress(self):
        pct = min(100.0, self.written * 100.0 / EXPECTED_TOTAL_SIZE)
        self.status_var.set(
            f"{pct:.0f}%    ·    {human(self.written)} / {human(EXPECTED_TOTAL_SIZE)}")
        self._draw_progress()

    def _status(self, text, color=MUTED):
        self.status_lbl.configure(fg=color)
        self.status_var.set(text)

    def _on_success(self, final_path):
        self.running = False
        self.written = EXPECTED_TOTAL_SIZE
        self._draw_progress()
        self._status("Instalacion completa.", A1)
        self.dest_final = final_path
        self._set_buttons(True)
        self.install_btn.reconfigure(text="Abrir carpeta", command=self._open_folder)
        messagebox.showinfo(APP_TITLE, f"¡Listo!\n\n{FINAL_NAME} se instalo en:\n{final_path}")

    def _fail(self, msg):
        def apply():
            self.running = False
            self.written = 0
            self._node_r = self.NODE_MIN
            self._draw_progress()
            self._status("Fallo la instalacion.", ERROR)
            self._set_buttons(True)
            self.install_btn.reconfigure(text="Instalar", command=self.start_install)
            messagebox.showerror(APP_TITLE, msg)
        self._ui(apply)

    def _open_folder(self):
        try:
            os.startfile(self.dest)   # solo Windows; el instalador final corre en Windows
        except Exception:
            pass

    def _set_buttons(self, enabled):
        self.install_btn.set_enabled(enabled)
        self.browse_btn.set_enabled(enabled)

    @staticmethod
    def _safe_remove(path):
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass

    def _center(self):
        self.root.update_idletasks()
        w, h = 560, 474
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{w}x{h}+{(sw - w) // 2}+{(sh - h) // 3}")


def main():
    root = tk.Tk()
    Installer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
