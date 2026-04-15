"""
admin_ui.py — Standalone Admin UI for the Egyptian ID pipeline.
Run with:  python admin_ui.py
No model needs to be running first.
"""
import json
import logging
import multiprocessing as mp
import os
import shutil
import sys
import threading
import tkinter as tk
import traceback as tb_module
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import cv2
from PIL import Image, ImageDraw, ImageTk
import template_fields as tf
import card_detect

# ── Logging setup ─────────────────────────────────────────────────────────────
LOG_FILE = Path(__file__).resolve().parent / "admin_ui.log"
logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] [%(levelname)7s] %(funcName)s — %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(str(LOG_FILE), encoding="utf-8", mode="a"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("admin_ui")

HERE = Path(__file__).resolve().parent
CONFIG_PATH    = HERE / "id_template_config.json"
TEMPLATES_DIR  = HERE / "templates"
FEATURES_DIR   = HERE / "features"
ADMIN_CFG_PATH = HERE / "admin_config.json"

FRONT_TEMPLATE = TEMPLATES_DIR / "front_template.png"
BACK_TEMPLATE  = TEMPLATES_DIR / "back_template.png"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

STAGES = [
    ("feature_front",  "Feature Validation — Front"),
    ("feature_back",   "Feature Validation — Back"),
    ("alignment",      "Alignment"),
    ("ocr",            "OCR (Text Extraction)"),
    ("liveness",       "Liveness Check"),
    ("face_match",     "Face Match"),
]

ROI_COLORS = [
    "#e74c3c", "#2980b9", "#27ae60", "#f39c12",
    "#8e44ad", "#16a085", "#d35400", "#2c3e50",
]

DISPLAY_W = 820
DISPLAY_H = 520


# ── helpers ───────────────────────────────────────────────────────────────────

def load_config() -> dict:
    log.debug("Loading config from %s", CONFIG_PATH)
    if CONFIG_PATH.exists():
        cfg = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        log.info("Config loaded — %d sides", len(cfg))
        return cfg
    log.warning("Config file not found, using defaults")
    return {"front": {"roi_pad_px": 4, "ocr_rois": {}}, "back": {"roi_pad_px": 4, "ocr_rois": {}}}


def save_config(cfg: dict):
    log.info("Saving config to %s", CONFIG_PATH)
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    log.debug("Config saved successfully")


def load_admin_cfg() -> dict:
    log.debug("Loading admin config from %s", ADMIN_CFG_PATH)
    if ADMIN_CFG_PATH.exists():
        cfg = json.loads(ADMIN_CFG_PATH.read_text(encoding="utf-8"))
        log.info("Admin config loaded — stages: %s", cfg)
        return cfg
    log.warning("Admin config not found, all stages enabled by default")
    return {k: True for k, _ in STAGES}


def save_admin_cfg(cfg: dict):
    log.info("Saving admin config: %s", cfg)
    ADMIN_CFG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def cv2_to_pil(bgr):
    log.debug("cv2_to_pil: shape=%s dtype=%s", bgr.shape, bgr.dtype)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def load_pil(path: Path) -> Image.Image:
    log.debug("load_pil: %s", path)
    img = tf.read_image(path)
    if img is None:
        log.error("load_pil FAILED: Cannot read %s", path)
        raise FileNotFoundError(f"Cannot read {path}")
    log.debug("load_pil OK: %s shape=%s", path, img.shape)
    return cv2_to_pil(img)


def fit_image(pil_img: Image.Image, max_w=DISPLAY_W, max_h=DISPLAY_H) -> Image.Image:
    w, h = pil_img.size
    scale = min(max_w / w, max_h / h, 1.0)
    new_w, new_h = int(w * scale), int(h * scale)
    log.debug("fit_image: %dx%d -> %dx%d (scale=%.3f)", w, h, new_w, new_h, scale)
    return pil_img.resize((new_w, new_h), Image.LANCZOS), scale


# ── Tab 1: Template Manager ────────────────────────────────────────────────────

class TemplateTab(tk.Frame):
    """
    Template & ROI editor.
    - Mouse drag on canvas  → draw a new box (named via prompt)
    - Click inside a box    → select it (highlights in list, shows coords)
    - Delete button in list → remove ROI
    - Coord entries         → full-precision, live preview on canvas
    """

    def __init__(self, parent):
        log.info("TemplateTab.__init__")
        super().__init__(parent, bg="#1e1e2e")
        self._side      = tk.StringVar(value="front")
        self._cfg       = load_config()
        self._pil_orig  = None          # original full-res PIL image
        self._disp_w    = 0             # displayed image width on canvas
        self._disp_h    = 0             # displayed image height on canvas
        self._scale     = 1.0
        self._tk_img    = None
        self._roi_entries  = {}         # name -> (x1_var, y1_var, x2_var, y2_var)
        self._roi_rows     = {}         # name -> row Frame (for highlight)
        self._selected     = None       # currently selected ROI name
        # drag state
        self._drag_start   = None       # (cx, cy) canvas coords
        self._drag_rect_id = None       # canvas rectangle id while dragging
        self._build()
        self._load_template()

    # ── layout ────────────────────────────────────────────────────────────────

    def _build(self):
        # ── top bar ──
        top = tk.Frame(self, bg="#1e1e2e")
        top.pack(fill="x", pady=6, padx=8)

        tk.Label(top, text="Side:", bg="#1e1e2e", fg="white",
                 font=("Segoe UI", 10)).pack(side="left")
        for val, txt in [("front", "Front"), ("back", "Back")]:
            tk.Radiobutton(top, text=txt, variable=self._side, value=val,
                           bg="#1e1e2e", fg="white", selectcolor="#313244",
                           activebackground="#1e1e2e", font=("Segoe UI", 10),
                           command=self._on_side_change).pack(side="left", padx=4)

        tk.Button(top, text="Upload New Template", command=self._upload_template,
                  bg="#7c3aed", fg="white", relief="flat", padx=8).pack(side="left", padx=12)

        hint = tk.Label(top, text="Drag on image to draw a box  |  Click box to select",
                        bg="#1e1e2e", fg="#6c7086", font=("Segoe UI", 8))
        hint.pack(side="left", padx=8)

        # ── main area ──
        mid = tk.Frame(self, bg="#1e1e2e")
        mid.pack(fill="both", expand=True, padx=8, pady=4)

        # ── canvas (with scrollbars for large templates) ──
        canvas_outer = tk.Frame(mid, bg="#313244", bd=1, relief="sunken")
        canvas_outer.pack(side="left", fill="both", expand=True)

        self._canvas = tk.Canvas(canvas_outer, bg="#181825", cursor="crosshair",
                                 scrollregion=(0, 0, DISPLAY_W, DISPLAY_H))
        hbar = ttk.Scrollbar(canvas_outer, orient="horizontal",
                             command=self._canvas.xview)
        vbar = ttk.Scrollbar(canvas_outer, orient="vertical",
                             command=self._canvas.yview)
        self._canvas.configure(xscrollcommand=hbar.set, yscrollcommand=vbar.set)
        hbar.pack(side="bottom", fill="x")
        vbar.pack(side="right",  fill="y")
        self._canvas.pack(fill="both", expand=True)

        self._canvas.bind("<ButtonPress-1>",   self._on_press)
        self._canvas.bind("<B1-Motion>",       self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)

        # ── ROI panel (right side) ──
        roi_outer = tk.Frame(mid, bg="#1e1e2e", width=340)
        roi_outer.pack(side="right", fill="y", padx=(8, 0))
        roi_outer.pack_propagate(False)

        hdr = tk.Frame(roi_outer, bg="#1e1e2e")
        hdr.pack(fill="x")
        tk.Label(hdr, text="ROI Regions", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 11, "bold")).pack(side="left")

        # scrollable list
        list_frame = tk.Frame(roi_outer, bg="#1e1e2e")
        list_frame.pack(fill="both", expand=True, pady=4)

        self._list_canvas = tk.Canvas(list_frame, bg="#1e1e2e", highlightthickness=0)
        sb = ttk.Scrollbar(list_frame, orient="vertical",
                           command=self._list_canvas.yview)
        self._roi_inner = tk.Frame(self._list_canvas, bg="#1e1e2e")
        self._roi_inner.bind("<Configure>", lambda e: self._list_canvas.configure(
            scrollregion=self._list_canvas.bbox("all")))
        self._list_canvas.create_window((0, 0), window=self._roi_inner, anchor="nw")
        self._list_canvas.configure(yscrollcommand=sb.set)
        self._list_canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # coord display for selected ROI
        coord_frame = tk.Frame(roi_outer, bg="#252535", bd=1, relief="sunken", pady=6, padx=6)
        coord_frame.pack(fill="x", pady=4)
        tk.Label(coord_frame, text="Selected ROI coordinates",
                 bg="#252535", fg="#a6adc8", font=("Segoe UI", 8)).grid(
                     row=0, column=0, columnspan=4, sticky="w", pady=(0, 4))
        self._coord_vars = []
        for col, lbl in enumerate(["x1", "y1", "x2", "y2"]):
            tk.Label(coord_frame, text=lbl, bg="#252535", fg="#cdd6f4",
                     font=("Segoe UI", 9, "bold")).grid(row=1, column=col, padx=4)
            v = tk.StringVar()
            v.trace_add("write", self._on_coord_panel_change)
            e = tk.Entry(coord_frame, textvariable=v, width=10,
                         bg="#1e1e2e", fg="white", insertbackground="white",
                         font=("Consolas", 9))
            e.grid(row=2, column=col, padx=4, pady=2)
            self._coord_vars.append(v)

        # add-roi row
        add_frame = tk.Frame(roi_outer, bg="#1e1e2e")
        add_frame.pack(fill="x", pady=2)
        tk.Label(add_frame, text="New name:", bg="#1e1e2e", fg="white",
                 font=("Segoe UI", 9)).pack(side="left")
        self._new_roi_name = tk.Entry(add_frame, width=14, bg="#313244", fg="white",
                                      insertbackground="white")
        self._new_roi_name.pack(side="left", padx=4)
        tk.Button(add_frame, text="Add", command=self._add_roi,
                  bg="#22c55e", fg="white", relief="flat", padx=6).pack(side="left")

        tk.Button(roi_outer, text="💾  Save Config", command=self._save,
                  bg="#3b82f6", fg="white", relief="flat", padx=10,
                  font=("Segoe UI", 10, "bold")).pack(fill="x", pady=6)

    # ── template loading ──────────────────────────────────────────────────────

    def _on_side_change(self):
        log.info("TemplateTab._on_side_change -> %s", self._side.get())
        self._selected = None
        self._load_template()

    def _load_template(self):
        side = self._side.get()
        tpl_path = FRONT_TEMPLATE if side == "front" else BACK_TEMPLATE
        log.info("TemplateTab._load_template: side=%s path=%s", side, tpl_path)
        try:
            self._pil_orig = load_pil(tpl_path)
            log.info("Template loaded OK: %s", tpl_path)
        except Exception as e:
            log.error("Template load FAILED: %s", e, exc_info=True)
            messagebox.showerror("Error", f"Cannot load template:\n{e}")
            return
        self._refresh_roi_panel()
        self._redraw()

    def _upload_template(self):
        log.info("TemplateTab._upload_template")
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp")])
        if not path:
            log.debug("Upload cancelled")
            return
        side = self._side.get()
        dest = FRONT_TEMPLATE if side == "front" else BACK_TEMPLATE
        log.info("Copying template %s -> %s", path, dest)
        TEMPLATES_DIR.mkdir(exist_ok=True)
        shutil.copy2(path, dest)
        self._load_template()
        messagebox.showinfo("Saved", f"Template saved to:\n{dest}")

    # ── ROI list panel ────────────────────────────────────────────────────────

    def _refresh_roi_panel(self):
        side = self._side.get()
        rois = self._cfg.get(side, {}).get("ocr_rois", {})
        log.debug("TemplateTab._refresh_roi_panel: side=%s rois=%d", side, len(rois))
        for w in self._roi_inner.winfo_children():
            w.destroy()
        self._roi_entries.clear()
        self._roi_rows.clear()
        for idx, (name, box) in enumerate(rois.items()):
            color = ROI_COLORS[idx % len(ROI_COLORS)]
            self._add_roi_row(name, box, color)

    def _add_roi_row(self, name: str, box: list, color: str = "#e74c3c"):
        bg = "#3a3a5c" if name == self._selected else "#313244"
        row = tk.Frame(self._roi_inner, bg=bg, pady=4, padx=6)
        row.pack(fill="x", pady=2, padx=2)
        self._roi_rows[name] = row

        # colour swatch + name
        tk.Label(row, text="■", fg=color, bg=bg,
                 font=("Segoe UI", 12)).grid(row=0, column=0, padx=(0, 4))
        name_lbl = tk.Label(row, text=name, bg=bg, fg="#cdd6f4",
                            font=("Segoe UI", 9, "bold"), anchor="w", width=14)
        name_lbl.grid(row=0, column=1, sticky="w")

        # bind click on name/row to select
        for widget in (row, name_lbl):
            widget.bind("<Button-1>", lambda e, n=name: self._select_roi(n))

        # full-precision coordinate entries — 2×2 grid
        vars_ = []
        defaults = box if len(box) == 4 else [0.0, 0.0, 1.0, 1.0]
        labels = ["x1", "y1", "x2", "y2"]
        for i, lbl in enumerate(labels):
            r, c = divmod(i, 2)
            v = tk.StringVar(value=f"{defaults[i]:.6f}")
            v.trace_add("write", lambda *_, n=name: self._on_entry_change(n))
            tk.Label(row, text=lbl, bg=bg, fg="#a6adc8",
                     font=("Segoe UI", 8)).grid(row=r + 1, column=c * 2,
                                                 padx=(4, 0), sticky="e")
            e = tk.Entry(row, textvariable=v, width=10, bg="#1e1e2e",
                         fg="white", insertbackground="white",
                         font=("Consolas", 8))
            e.grid(row=r + 1, column=c * 2 + 1, padx=(0, 6), pady=1)
            vars_.append(v)

        # delete button
        tk.Button(row, text="🗑 Delete",
                  command=lambda n=name: self._delete_roi(n),
                  bg="#f38ba8", fg="white", relief="flat",
                  font=("Segoe UI", 8)).grid(row=0, column=2, padx=4)

        self._roi_entries[name] = tuple(vars_)

    def _select_roi(self, name: str):
        log.debug("TemplateTab._select_roi: %s", name)
        self._selected = name
        self._refresh_roi_panel()
        # populate coord panel
        side = self._side.get()
        box  = self._cfg.get(side, {}).get("ocr_rois", {}).get(name, [0, 0, 1, 1])
        self._updating_panel = True
        for i, v in enumerate(self._coord_vars):
            v.set(f"{box[i]:.6f}")
        self._updating_panel = False
        self._redraw()

    def _on_entry_change(self, name: str):
        """Called when a coord entry in the ROI list row changes."""
        self._sync_roi_to_cfg(name)
        # keep coord panel in sync if this is the selected ROI
        if name == self._selected:
            side = self._side.get()
            box  = self._cfg.get(side, {}).get("ocr_rois", {}).get(name)
            if box:
                self._updating_panel = True
                for i, v in enumerate(self._coord_vars):
                    v.set(f"{box[i]:.6f}")
                self._updating_panel = False
        self._redraw()

    _updating_panel = False  # guard against circular trace

    def _on_coord_panel_change(self, *_):
        """Called when the bottom coord-panel entries change."""
        if self._updating_panel or not self._selected:
            return
        name = self._selected
        if name not in self._roi_entries:
            return
        try:
            vals = [float(v.get()) for v in self._coord_vars]
        except ValueError:
            return
        side = self._side.get()
        self._cfg.setdefault(side, {}).setdefault("ocr_rois", {})[name] = vals
        # push to per-row entries
        self._updating_panel = True
        for i, v in enumerate(self._roi_entries[name]):
            v.set(f"{vals[i]:.6f}")
        self._updating_panel = False
        self._redraw()

    def _sync_roi_to_cfg(self, name: str):
        side = self._side.get()
        if name not in self._roi_entries:
            return
        try:
            vals = [float(v.get()) for v in self._roi_entries[name]]
            self._cfg.setdefault(side, {}).setdefault("ocr_rois", {})[name] = vals
        except ValueError:
            pass

    def _add_roi(self):
        name = self._new_roi_name.get().strip()
        log.info("TemplateTab._add_roi: name=%r", name)
        if not name:
            messagebox.showwarning("Name required", "Enter a name for the new ROI.")
            return
        side = self._side.get()
        rois = self._cfg.setdefault(side, {}).setdefault("ocr_rois", {})
        if name in rois:
            messagebox.showwarning("Duplicate", f"ROI '{name}' already exists.")
            return
        rois[name] = [0.1, 0.1, 0.9, 0.2]
        self._new_roi_name.delete(0, "end")
        self._selected = name
        self._refresh_roi_panel()
        self._redraw()

    def _delete_roi(self, name: str):
        log.info("TemplateTab._delete_roi: %s", name)
        side = self._side.get()
        rois = self._cfg.get(side, {}).get("ocr_rois", {})
        if name in rois:
            del rois[name]
        if self._selected == name:
            self._selected = None
            for v in self._coord_vars:
                v.set("")
        self._refresh_roi_panel()
        self._redraw()

    # ── mouse draw ────────────────────────────────────────────────────────────

    def _canvas_to_norm(self, cx: int, cy: int):
        """Convert canvas pixel coords to normalised [0..1] relative to image."""
        if self._disp_w == 0 or self._disp_h == 0:
            return 0.0, 0.0
        nx = max(0.0, min(1.0, cx / self._disp_w))
        ny = max(0.0, min(1.0, cy / self._disp_h))
        return nx, ny

    def _on_press(self, event):
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)

        # check if click is inside an existing ROI → select it
        hit = self._hit_test(cx, cy)
        if hit:
            self._select_roi(hit)
            self._drag_start = None
            return

        self._drag_start  = (cx, cy)
        self._drag_rect_id = None

    def _on_drag(self, event):
        if self._drag_start is None:
            return
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)
        x0, y0 = self._drag_start
        if self._drag_rect_id:
            self._canvas.delete(self._drag_rect_id)
        self._drag_rect_id = self._canvas.create_rectangle(
            x0, y0, cx, cy, outline="#ffffff", width=2, dash=(4, 2))

    def _on_release(self, event):
        if self._drag_start is None:
            return
        cx = self._canvas.canvasx(event.x)
        cy = self._canvas.canvasy(event.y)
        x0, y0 = self._drag_start
        self._drag_start = None
        if self._drag_rect_id:
            self._canvas.delete(self._drag_rect_id)
            self._drag_rect_id = None

        # ignore tiny drags (clicks)
        if abs(cx - x0) < 5 or abs(cy - y0) < 5:
            return

        # normalise coords (ensure x1<x2, y1<y2)
        nx1, ny1 = self._canvas_to_norm(min(x0, cx), min(y0, cy))
        nx2, ny2 = self._canvas_to_norm(max(x0, cx), max(y0, cy))

        # ask for name
        name = self._ask_roi_name()
        if not name:
            return

        side = self._side.get()
        rois = self._cfg.setdefault(side, {}).setdefault("ocr_rois", {})
        rois[name] = [round(nx1, 6), round(ny1, 6), round(nx2, 6), round(ny2, 6)]
        self._selected = name
        self._refresh_roi_panel()
        self._redraw()

    def _ask_roi_name(self) -> str:
        """Popup dialog to enter the ROI name."""
        pre = self._new_roi_name.get().strip()
        if pre:
            side = self._side.get()
            rois = self._cfg.get(side, {}).get("ocr_rois", {})
            if pre not in rois:
                self._new_roi_name.delete(0, "end")
                return pre

        dlg = tk.Toplevel(self)
        dlg.title("ROI Name")
        dlg.configure(bg="#1e1e2e")
        dlg.resizable(False, False)
        dlg.grab_set()
        tk.Label(dlg, text="Enter a name for this ROI:",
                 bg="#1e1e2e", fg="white",
                 font=("Segoe UI", 10)).pack(padx=16, pady=(12, 4))
        var = tk.StringVar()
        e = tk.Entry(dlg, textvariable=var, width=22, bg="#313244",
                     fg="white", insertbackground="white",
                     font=("Segoe UI", 10))
        e.pack(padx=16, pady=4)
        e.focus_set()
        result = []

        def ok(_=None):
            n = var.get().strip()
            if n:
                result.append(n)
            dlg.destroy()

        e.bind("<Return>", ok)
        tk.Button(dlg, text="OK", command=ok,
                  bg="#3b82f6", fg="white", relief="flat", padx=12).pack(pady=(4, 12))
        self.wait_window(dlg)
        return result[0] if result else ""

    def _hit_test(self, cx: float, cy: float) -> str:
        """Return the name of the ROI the canvas point falls inside, or ''."""
        if self._disp_w == 0 or self._disp_h == 0:
            return ""
        side = self._side.get()
        rois = self._cfg.get(side, {}).get("ocr_rois", {})
        # iterate in reverse so topmost (last drawn) wins
        for name, box in reversed(list(rois.items())):
            if len(box) != 4:
                continue
            bx1 = box[0] * self._disp_w
            by1 = box[1] * self._disp_h
            bx2 = box[2] * self._disp_w
            by2 = box[3] * self._disp_h
            if bx1 <= cx <= bx2 and by1 <= cy <= by2:
                return name
        return ""

    # ── canvas redraw ─────────────────────────────────────────────────────────

    def _redraw(self):
        if self._pil_orig is None:
            return
        img_fit, self._scale = fit_image(self._pil_orig)
        self._disp_w, self._disp_h = img_fit.size

        draw = ImageDraw.Draw(img_fit)
        side = self._side.get()
        rois = self._cfg.get(side, {}).get("ocr_rois", {})

        for idx, (name, box) in enumerate(rois.items()):
            if len(box) != 4:
                continue
            color   = ROI_COLORS[idx % len(ROI_COLORS)]
            is_sel  = (name == self._selected)
            px1 = int(box[0] * self._disp_w)
            py1 = int(box[1] * self._disp_h)
            px2 = int(box[2] * self._disp_w)
            py2 = int(box[3] * self._disp_h)
            width = 3 if is_sel else 2
            draw.rectangle([px1, py1, px2, py2], outline=color, width=width)
            if is_sel:
                # draw a semi-transparent fill tint
                overlay = Image.new("RGBA", img_fit.size, (0, 0, 0, 0))
                od = ImageDraw.Draw(overlay)
                od.rectangle([px1, py1, px2, py2],
                              fill=(*self._hex_to_rgb(color), 40))
                img_fit = img_fit.convert("RGBA")
                img_fit = Image.alpha_composite(img_fit, overlay).convert("RGB")
                draw = ImageDraw.Draw(img_fit)
                draw.rectangle([px1, py1, px2, py2], outline=color, width=3)
            # label with name + normalised coords
            coord_txt = f"{name}  [{box[0]:.4f},{box[1]:.4f},{box[2]:.4f},{box[3]:.4f}]"
            draw.text((px1 + 3, py1 + 2), coord_txt, fill=color)

        self._tk_img = ImageTk.PhotoImage(img_fit)
        self._canvas.configure(
            scrollregion=(0, 0, self._disp_w, self._disp_h))
        self._canvas.delete("all")
        self._canvas.create_image(0, 0, anchor="nw", image=self._tk_img)

    @staticmethod
    def _hex_to_rgb(hex_color: str):
        h = hex_color.lstrip("#")
        return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

    # ── save ─────────────────────────────────────────────────────────────────

    def _save(self):
        log.info("TemplateTab._save")
        side = self._side.get()
        for name in list(self._roi_entries.keys()):
            self._sync_roi_to_cfg(name)
        save_config(self._cfg)
        messagebox.showinfo("Saved", f"Config saved to:\n{CONFIG_PATH}")


# ── Tab 2: Features Manager ────────────────────────────────────────────────────

class FeaturesTab(tk.Frame):
    def __init__(self, parent):
        log.info("FeaturesTab.__init__")
        super().__init__(parent, bg="#1e1e2e")
        self._side = tk.StringVar(value="front")
        self._build()
        self._refresh()

    def _build(self):
        top = tk.Frame(self, bg="#1e1e2e")
        top.pack(fill="x", pady=6, padx=8)

        tk.Label(top, text="Side:", bg="#1e1e2e", fg="white", font=("Segoe UI", 10)).pack(side="left")
        for val, txt in [("front", "Front"), ("back", "Back")]:
            tk.Radiobutton(top, text=txt, variable=self._side, value=val,
                           bg="#1e1e2e", fg="white", selectcolor="#313244",
                           activebackground="#1e1e2e", font=("Segoe UI", 10),
                           command=self._refresh).pack(side="left", padx=4)

        tk.Button(top, text="Upload Feature Image", command=self._upload,
                  bg="#7c3aed", fg="white", relief="flat", padx=8).pack(side="left", padx=12)

        mid = tk.Frame(self, bg="#1e1e2e")
        mid.pack(fill="both", expand=True, padx=8)

        # List
        list_frame = tk.Frame(mid, bg="#313244", bd=1, relief="sunken")
        list_frame.pack(side="left", fill="both", expand=False, pady=4)

        self._listbox = tk.Listbox(list_frame, bg="#1e1e2e", fg="#cdd6f4",
                                   selectbackground="#3b82f6", width=28,
                                   font=("Segoe UI", 9))
        self._listbox.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(list_frame, orient="vertical", command=self._listbox.yview)
        sb.pack(side="right", fill="y")
        self._listbox.config(yscrollcommand=sb.set)
        self._listbox.bind("<<ListboxSelect>>", self._on_select)

        btn_frame = tk.Frame(mid, bg="#1e1e2e")
        btn_frame.pack(side="left", fill="y", padx=8)
        tk.Button(btn_frame, text="🗑 Delete Selected", command=self._delete,
                  bg="#f38ba8", fg="white", relief="flat", padx=8).pack(pady=4)

        # Preview
        preview_frame = tk.Frame(mid, bg="#181825", bd=1, relief="sunken")
        preview_frame.pack(side="left", fill="both", expand=True, pady=4)
        self._preview_canvas = tk.Canvas(preview_frame, bg="#181825",
                                         width=400, height=DISPLAY_H)
        self._preview_canvas.pack(fill="both", expand=True)
        self._tk_prev = None

    def _feature_dir(self) -> Path:
        return FEATURES_DIR / self._side.get()

    def _refresh(self):
        d = self._feature_dir()
        log.debug("FeaturesTab._refresh: dir=%s", d)
        d.mkdir(parents=True, exist_ok=True)
        self._listbox.delete(0, "end")
        self._files = []
        for p in sorted(d.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                self._listbox.insert("end", p.name)
                self._files.append(p)
        log.debug("FeaturesTab._refresh: found %d images", len(self._files))
        self._preview_canvas.delete("all")

    def _on_select(self, _event=None):
        sel = self._listbox.curselection()
        if not sel:
            return
        path = self._files[sel[0]]
        try:
            pil = load_pil(path)
            fit, _ = fit_image(pil, max_w=400, max_h=DISPLAY_H)
            self._tk_prev = ImageTk.PhotoImage(fit)
            self._preview_canvas.config(width=fit.width, height=fit.height)
            self._preview_canvas.delete("all")
            self._preview_canvas.create_image(0, 0, anchor="nw", image=self._tk_prev)
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def _upload(self):
        log.info("FeaturesTab._upload")
        paths = filedialog.askopenfilenames(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.webp")])
        if not paths:
            return
        d = self._feature_dir()
        d.mkdir(parents=True, exist_ok=True)
        for p in paths:
            shutil.copy2(p, d / Path(p).name)
        self._refresh()

    def _delete(self):
        sel = self._listbox.curselection()
        if not sel:
            return
        path = self._files[sel[0]]
        log.info("FeaturesTab._delete: %s", path)
        if messagebox.askyesno("Delete", f"Delete {path.name}?"):
            path.unlink(missing_ok=True)
            self._refresh()


# ── Tab 3: Stage Toggles ──────────────────────────────────────────────────────

class StagesTab(tk.Frame):
    def __init__(self, parent):
        log.info("StagesTab.__init__")
        super().__init__(parent, bg="#1e1e2e")
        self._admin_cfg = load_admin_cfg()
        self._vars = {}
        self._build()

    def _build(self):
        tk.Label(self, text="Enable / Disable Pipeline Stages",
                 bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 13, "bold")).pack(pady=16)

        tk.Label(self, text="Disabled stages are skipped — pipeline continues to the next enabled stage.",
                 bg="#1e1e2e", fg="#a6adc8", font=("Segoe UI", 9)).pack(pady=(0, 12))

        for key, label in STAGES:
            var = tk.BooleanVar(value=self._admin_cfg.get(key, True))
            self._vars[key] = var
            row = tk.Frame(self, bg="#313244", pady=8, padx=16)
            row.pack(fill="x", padx=40, pady=4)
            cb = tk.Checkbutton(row, text=label, variable=var,
                                 bg="#313244", fg="white", selectcolor="#1e1e2e",
                                 activebackground="#313244",
                                 font=("Segoe UI", 11), anchor="w")
            cb.pack(side="left")

        tk.Button(self, text="💾 Save Stage Settings", command=self._save,
                  bg="#3b82f6", fg="white", relief="flat", padx=14,
                  font=("Segoe UI", 10, "bold")).pack(pady=20)

    def _save(self):
        log.info("StagesTab._save")
        for key, var in self._vars.items():
            self._admin_cfg[key] = var.get()
        save_admin_cfg(self._admin_cfg)
        messagebox.showinfo("Saved", "Stage settings saved.")

    def get_enabled(self) -> dict:
        return {key: var.get() for key, var in self._vars.items()}


# ── OCR subprocess worker (module-level for pickling on Windows) ──────────────

def _ocr_in_process(front_crops, back_crops, result_queue):
    """Run OCR in a separate process to isolate native PaddlePaddle crashes."""
    try:
        import arabic_ocr as ao
        fields = ao.extract_fields_from_crops(
            front_crops=front_crops, back_crops=back_crops)
        safe = {k: v for k, v in fields.items() if k != "raw"}
        result_queue.put(("ok", safe))
    except Exception:
        import traceback
        result_queue.put(("error", traceback.format_exc()))


# ── Tab 4: Run Pipeline ────────────────────────────────────────────────────────

class RunTab(tk.Frame):
    def __init__(self, parent, stages_tab: StagesTab, visual_tab: "VisualResultsTab"):
        log.info("RunTab.__init__")
        super().__init__(parent, bg="#1e1e2e")
        self._stages_tab = stages_tab
        self._visual_tab = visual_tab
        self._paths = {}
        self._alive = True
        self._build()

    def _build(self):
        tk.Label(self, text="Run Pipeline", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 13, "bold")).pack(pady=12)

        inputs = [
            ("selfie",       "Selfie Image (optional)"),
            ("front_image",  "Front ID Image (optional)"),
            ("back_image",   "Back ID Image (optional)"),
        ]

        for key, label in inputs:
            row = tk.Frame(self, bg="#1e1e2e")
            row.pack(fill="x", padx=40, pady=4)
            tk.Label(row, text=label, bg="#1e1e2e", fg="#cdd6f4",
                     font=("Segoe UI", 10), width=28, anchor="w").pack(side="left")
            var = tk.StringVar()
            self._paths[key] = var
            tk.Entry(row, textvariable=var, width=40, bg="#313244", fg="white",
                     insertbackground="white").pack(side="left", padx=6)
            tk.Button(row, text="Browse…",
                      command=lambda k=key: self._browse(k),
                      bg="#7c3aed", fg="white", relief="flat", padx=6).pack(side="left")

        tk.Button(self, text="▶  Run Pipeline", command=self._run,
                  bg="#22c55e", fg="white", relief="flat", padx=20,
                  font=("Segoe UI", 11, "bold")).pack(pady=14)

        self._status = tk.Label(self, text="", bg="#1e1e2e", fg="#f9e2af",
                                font=("Segoe UI", 10))
        self._status.pack()

        out_frame = tk.Frame(self, bg="#1e1e2e")
        out_frame.pack(fill="both", expand=True, padx=40, pady=8)
        tk.Label(out_frame, text="Result:", bg="#1e1e2e", fg="#a6adc8",
                 font=("Segoe UI", 9)).pack(anchor="w")
        self._output = scrolledtext.ScrolledText(out_frame, bg="#181825", fg="#cdd6f4",
                                                 font=("Consolas", 9), state="disabled")
        self._output.pack(fill="both", expand=True)

    def _browse(self, key: str):
        log.debug("RunTab._browse: key=%s", key)
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if path:
            log.info("RunTab._browse: %s = %s", key, path)
            self._paths[key].set(path)

    def _run(self):
        log.info("RunTab._run: starting pipeline thread")
        self._status.config(text="Running…", fg="#f9e2af")
        self._set_output("")
        threading.Thread(target=self._run_thread, daemon=True).start()

    def _run_thread(self):
        try:
            log.info("=" * 60)
            log.info("PIPELINE RUN STARTED")
            log.info("=" * 60)
            sys.path.insert(0, str(HERE))

            log.debug("Importing arabic_ocr...")
            import arabic_ocr as ao
            log.debug("Importing face_match_top4...")
            import face_match_top4 as fm
            log.debug("Importing feature_pipeline...")
            import feature_pipeline as fp
            log.debug("Importing liveness_gate...")
            import liveness_gate as lg
            log.debug("Importing template_fields...")
            import template_fields as tf
            log.info("All imports OK")

            enabled = self._stages_tab.get_enabled()
            log.info("Enabled stages: %s", enabled)
            cfg = load_config()

            selfie      = self._paths["selfie"].get().strip() or None
            front_img   = self._paths["front_image"].get().strip() or None
            back_img    = self._paths["back_image"].get().strip() or None
            log.info("Inputs: selfie=%s  front=%s  back=%s", selfie, front_img, back_img)

            result = {}

            # ── Feature validation ────────────────────────────────────────────
            if enabled.get("feature_front") or enabled.get("feature_back"):
                log.info("[STAGE] Feature validation")
                feat_result = {"front": None, "back": None}
                if enabled.get("feature_front") and front_img:
                    front_dir = FEATURES_DIR / "front"
                    if front_dir.exists():
                        try:
                            log.info("  Running feature_front: %s", front_img)
                            ok, logs = fp.validate_subject_with_top_models(
                                subject_path=front_img,
                                features_dir=str(front_dir),
                                include_deep=False, top_k=4, side_name="front")
                            feat_result["front"] = {"passed": bool(ok), "logs": logs}
                            log.info("  feature_front result: passed=%s", ok)
                        except Exception as e:
                            log.error("  feature_front ERROR: %s", e, exc_info=True)
                            feat_result["front"] = {"passed": False, "error": str(e)}
                if enabled.get("feature_back") and back_img:
                    back_dir = FEATURES_DIR / "back"
                    if back_dir.exists():
                        try:
                            log.info("  Running feature_back: %s", back_img)
                            ok, logs = fp.validate_subject_with_top_models(
                                subject_path=back_img,
                                features_dir=str(back_dir),
                                include_deep=False, top_k=4, side_name="back")
                            feat_result["back"] = {"passed": bool(ok), "logs": logs}
                            log.info("  feature_back result: passed=%s", ok)
                        except Exception as e:
                            log.error("  feature_back ERROR: %s", e, exc_info=True)
                            feat_result["back"] = {"passed": False, "error": str(e)}
                result["feature_validation"] = feat_result
            else:
                log.info("[STAGE] Feature validation — SKIPPED")

            # ── Alignment + artifact extraction ──────────────────────────────
            front_art = back_art = None
            if enabled.get("alignment"):
                log.info("[STAGE] Alignment")
                if front_img and FRONT_TEMPLATE.exists():
                    try:
                        log.info("  Aligning front: %s", front_img)
                        aligned_f, info_f = tf.align_card_to_template(
                            front_img, str(FRONT_TEMPLATE))
                        log.info("  Front aligned OK — rotation=%s score=%s",
                                 info_f.get('rotation'), info_f.get('score'))
                        front_art = tf.extract_side_artifacts(
                            aligned_f,
                            cfg,
                            "front",
                            subject_image=front_img,
                            align_info=info_f,
                        )
                        log.info("  Front artifacts: crops=%s photo=%s",
                                 list(front_art.get('crops', {}).keys()) if front_art else None,
                                 front_art.get('photo') is not None if front_art else None)
                        result["alignment_front"] = info_f
                    except Exception as e:
                        log.error("  Front alignment ERROR: %s", e, exc_info=True)
                        result["alignment_front"] = {"error": str(e)}
                if back_img and BACK_TEMPLATE.exists():
                    try:
                        log.info("  Aligning back: %s", back_img)
                        aligned_b, info_b = tf.align_card_to_template(
                            back_img, str(BACK_TEMPLATE))
                        log.info("  Back aligned OK — rotation=%s score=%s",
                                 info_b.get('rotation'), info_b.get('score'))
                        back_art = tf.extract_side_artifacts(
                            aligned_b,
                            cfg,
                            "back",
                            subject_image=back_img,
                            align_info=info_b,
                        )
                        log.info("  Back artifacts: crops=%s",
                                 list(back_art.get('crops', {}).keys()) if back_art else None)
                        result["alignment_back"] = info_b
                    except Exception as e:
                        log.error("  Back alignment ERROR: %s", e, exc_info=True)
                        result["alignment_back"] = {"error": str(e)}
            else:
                log.info("[STAGE] Alignment — SKIPPED")

            # ── OCR (runs in subprocess to survive native crashes) ────────────
            if enabled.get("ocr"):
                log.info("[STAGE] OCR (subprocess)")
                if front_art or back_art:
                    fc = front_art.get("crops", {}) if front_art else {}
                    bc = back_art.get("crops", {}) if back_art else None
                    log.info("  OCR front_crops=%s back_crops=%s",
                             list(fc.keys()), list(bc.keys()) if bc else None)
                    try:
                        q = mp.Queue()
                        p = mp.Process(target=_ocr_in_process, args=(fc, bc, q))
                        p.start()
                        log.info("  OCR subprocess started (pid=%s)", p.pid)
                        p.join(timeout=300)  # 5 minute timeout
                        if p.is_alive():
                            log.error("  OCR subprocess TIMED OUT — killing")
                            p.terminate()
                            p.join(5)
                            result["ocr"] = {"error": "OCR process timed out after 300 seconds"}
                        elif p.exitcode != 0:
                            log.error("  OCR subprocess CRASHED with exit code %s", p.exitcode)
                            result["ocr"] = {"error": f"OCR process crashed (exit code {p.exitcode}). This is a known PaddlePaddle issue."}
                        else:
                            try:
                                status, data = q.get_nowait()
                                if status == "ok":
                                    result["ocr"] = data
                                    log.info("  OCR result keys: %s", list(data.keys()))
                                else:
                                    log.error("  OCR returned error: %s", data[:500])
                                    result["ocr"] = {"error": data}
                            except Exception:
                                log.error("  OCR subprocess produced no output")
                                result["ocr"] = {"error": "OCR process produced no output"}
                    except Exception as e:
                        log.error("  OCR subprocess ERROR: %s", e, exc_info=True)
                        result["ocr"] = {"error": str(e)}
                else:
                    log.warning("  OCR skipped — no artifacts")
                    result["ocr"] = {"skipped": "No aligned artifacts available (alignment stage disabled or no images)"}
            else:
                log.info("[STAGE] OCR — SKIPPED")

            # ── Face match ────────────────────────────────────────────────────
            if enabled.get("face_match") and selfie and front_art:
                log.info("[STAGE] Face match")
                photo_crop = front_art.get("photo")
                if photo_crop is not None and getattr(photo_crop, "size", 0) > 0:
                    try:
                        log.info("  Running face match: selfie=%s", selfie)
                        passed, logs = fm.verify_top4_from_array(
                            selfie_path=selfie, target_face_bgr=photo_crop)
                        result["face_match"] = {"passed": bool(passed), "logs": logs}
                        log.info("  Face match result: passed=%s", passed)
                    except Exception as e:
                        log.error("  Face match ERROR: %s", e, exc_info=True)
                        result["face_match"] = {"passed": False, "error": str(e)}
                else:
                    log.warning("  Face match skipped — no photo crop")
                    result["face_match"] = {"skipped": "No photo crop found in ID"}
            else:
                log.info("[STAGE] Face match — SKIPPED (enabled=%s selfie=%s front_art=%s)",
                         enabled.get('face_match'), selfie is not None, front_art is not None)

            # ── Liveness ──────────────────────────────────────────────────────
            if enabled.get("liveness") and selfie:
                log.info("[STAGE] Liveness")
                try:
                    lv = lg.check_selfie_liveness(selfie_path=selfie)
                    result["liveness"] = lv
                    log.info("  Liveness result: %s", lv)
                except Exception as e:
                    log.error("  Liveness ERROR: %s", e, exc_info=True)
                    result["liveness"] = {"passed": False, "error": str(e)}
            else:
                log.info("[STAGE] Liveness — SKIPPED")

            log.info("Pipeline stages complete — building output")
            output = json.dumps(result, ensure_ascii=False, indent=2,
                                default=lambda o: str(o))
            self._safe_after(lambda: self._set_output(output))
            self._safe_after(lambda: self._status.config(text="\u2705 Done", fg="#a6e3a1"))

            # ── build visual sections ─────────────────────────────────────────
            log.info("Building visual sections")
            sections = []

            # 1. Card detection visuals
            detect_imgs = []
            if front_art and front_art.get("outline") is not None:
                detect_imgs.append(("Front \u2014 Card Outline", front_art["outline"]))
            if front_art and front_art.get("projection") is not None:
                detect_imgs.append(("Front \u2014 Projected Template ROIs", front_art["projection"]))
            if back_art and back_art.get("outline") is not None:
                detect_imgs.append(("Back \u2014 Card Outline", back_art["outline"]))
            if back_art and back_art.get("projection") is not None:
                detect_imgs.append(("Back \u2014 Projected Template ROIs", back_art["projection"]))
            if detect_imgs:
                sections.append(("\U0001f7e9  Card Detection", detect_imgs))

            # 2. Aligned images
            aligned_imgs = []
            if front_art and front_art.get("aligned") is not None:
                aligned_imgs.append(("Front \u2014 Aligned", front_art["aligned"]))
            if front_art and front_art.get("cleaned") is not None:
                aligned_imgs.append(("Front \u2014 Cleaned", front_art["cleaned"]))
            if back_art and back_art.get("aligned") is not None:
                aligned_imgs.append(("Back \u2014 Aligned", back_art["aligned"]))
            if back_art and back_art.get("cleaned") is not None:
                aligned_imgs.append(("Back \u2014 Cleaned", back_art["cleaned"]))
            if aligned_imgs:
                sections.append(("\U0001f5c2  Aligned Images", aligned_imgs))

            # 3. Cropped ROIs
            crop_imgs = []
            if front_art:
                if front_art.get("photo") is not None:
                    crop_imgs.append(("front \u00b7 photo", front_art["photo"]))
                for cname, crop in front_art.get("crops", {}).items():
                    crop_imgs.append((f"front \u00b7 {cname}", crop))
            if back_art:
                for cname, crop in back_art.get("crops", {}).items():
                    crop_imgs.append((f"back \u00b7 {cname}", crop))
            if crop_imgs:
                sections.append(("\u2702  Cropped ROIs", crop_imgs))

            # 4. Feature match visualizations
            log.debug("Building feature match visuals")
            feat_vis = []
            feat_val = result.get("feature_validation", {})
            for side_key in ("front", "back"):
                side_result = feat_val.get(side_key) or {}
                feat_logs = side_result.get("logs") or []
                src_path = front_img if side_key == "front" else back_img
                feat_dir  = FEATURES_DIR / side_key
                if src_path and feat_dir.exists():
                    subject_bgr = tf.read_image(src_path)
                    if subject_bgr is not None:
                        for flog in (feat_logs if isinstance(feat_logs, list) else []):
                            feat_path = None
                            if isinstance(flog, dict):
                                feat_path = flog.get("feature_path") or ""
                            if feat_path and Path(feat_path).exists():
                                feat_bgr = tf.read_image(feat_path)
                                if feat_bgr is not None:
                                    vis = _draw_match_vis(
                                        feat_bgr, subject_bgr,
                                        flog.get("model_name", ""),
                                        flog.get("inliers", 0),
                                        flog.get("score", 0.0),
                                        bool(flog.get("found", False)),
                                    )
                                    lbl = (f"{side_key} \u00b7 "
                                           f"{flog.get('model_name','')} \u00b7 "
                                           f"{Path(feat_path).name}")
                                    feat_vis.append((lbl, vis))
            if feat_vis:
                sections.append(("\U0001f50d  Feature Matching", feat_vis))

            log.info("Visual sections built: %d sections", len(sections))
            vis_sections = sections  # capture for lambda
            self._safe_after(lambda s=vis_sections: self._visual_tab.show_results(s))

            log.info("=" * 60)
            log.info("PIPELINE RUN COMPLETE")
            log.info("=" * 60)

        except Exception as e:
            log.error("PIPELINE RUN FAILED: %s", e, exc_info=True)
            msg = tb_module.format_exc()
            self._safe_after(lambda: self._set_output(msg))
            self._safe_after(lambda: self._status.config(text="\u274c Error", fg="#f38ba8"))

    def _safe_after(self, func):
        """Schedule func on the main thread, guarding against destroyed widgets."""
        try:
            if self._alive:
                self.after(0, func)
        except Exception as e:
            log.error("_safe_after failed: %s", e, exc_info=True)

    def _set_output(self, text: str):
        log.debug("RunTab._set_output: %d chars", len(text))
        try:
            self._output.config(state="normal")
            self._output.delete("1.0", "end")
            self._output.insert("end", text)
            self._output.config(state="disabled")
        except Exception as e:
            log.error("_set_output failed: %s", e, exc_info=True)


# ── feature match visualizer (standalone, no pipeline import needed) ──────────

def _draw_match_vis(feat_bgr, subject_bgr, model_name: str,
                    inliers: int, score: float, found: bool):
    """
    Side-by-side visualisation of feature image vs subject image.
    Draws ORB keypoints on both and overlays result text.
    Returns a BGR ndarray.
    """
    log.debug("_draw_match_vis: model=%s inliers=%d score=%.3f found=%s",
              model_name, inliers, score, found)
    try:
        TH = 300   # target height for each panel
        def _resize_h(img, h):
            ratio = h / max(img.shape[0], 1)
            return cv2.resize(img, (max(1, int(img.shape[1] * ratio)), h))

        f_rsz = _resize_h(feat_bgr,    TH)
        s_rsz = _resize_h(subject_bgr, TH)

        # draw ORB keypoints
        orb = cv2.ORB_create(nfeatures=500)
        kp_f = orb.detect(cv2.cvtColor(f_rsz, cv2.COLOR_BGR2GRAY), None)
        kp_s = orb.detect(cv2.cvtColor(s_rsz, cv2.COLOR_BGR2GRAY), None)
        f_kp = cv2.drawKeypoints(f_rsz, kp_f, None,
                                  color=(0, 230, 120),
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        s_kp = cv2.drawKeypoints(s_rsz, kp_s, None,
                                  color=(0, 180, 255),
                                  flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        # divider
        div = np.full((TH, 4, 3), (80, 80, 80), dtype=np.uint8)
        vis = np.hstack([f_kp, div, s_kp])

        # overlay text bar
        bar_h = 36
        bar = np.zeros((bar_h, vis.shape[1], 3), dtype=np.uint8)
        color = (60, 220, 60) if found else (60, 60, 220)
        status = "FOUND" if found else "NOT FOUND"
        txt = f"{model_name}  |  {status}  |  inliers={inliers}  score={score:.2f}"
        cv2.putText(bar, txt, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, color, 1, cv2.LINE_AA)
        vis = np.vstack([bar, vis])
    except Exception:
        # fallback: just return the feature image
        vis = feat_bgr.copy()
    return vis


# ── Tab 5: Visual Results ────────────────────────────────────────────────────

class VisualResultsTab(tk.Frame):
    """
    Displays after a pipeline run:
      - Aligned front / back images
      - All cropped ROI images (with label)
      - Feature matching visualizations (keypoint overlay)
    Each image is shown as a thumbnail in a scrollable grid.
    Click any thumbnail to enlarge it in a popup.
    """

    _THUMB_W = 220
    _THUMB_H = 140
    _COLS    = 4

    def __init__(self, parent):
        log.info("VisualResultsTab.__init__")
        super().__init__(parent, bg="#1e1e2e")
        self._tk_refs = []   # keep PhotoImage refs alive
        self._build()

    def _build(self):
        top = tk.Frame(self, bg="#1e1e2e")
        top.pack(fill="x", padx=8, pady=6)
        tk.Label(top, text="Visual Results", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 13, "bold")).pack(side="left")
        tk.Button(top, text="Clear", command=self.clear,
                  bg="#f38ba8", fg="white", relief="flat", padx=8).pack(side="right")

        self._status_lbl = tk.Label(self, text="No results yet — run the pipeline first.",
                                    bg="#1e1e2e", fg="#6c7086",
                                    font=("Segoe UI", 9))
        self._status_lbl.pack(pady=4)

        # scrollable canvas
        outer = tk.Frame(self, bg="#181825", bd=1, relief="sunken")
        outer.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self._scroll_canvas = tk.Canvas(outer, bg="#181825", highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical",
                            command=self._scroll_canvas.yview)
        hsb = ttk.Scrollbar(outer, orient="horizontal",
                            command=self._scroll_canvas.xview)
        self._inner = tk.Frame(self._scroll_canvas, bg="#181825")
        self._inner.bind("<Configure>", lambda e: self._scroll_canvas.configure(
            scrollregion=self._scroll_canvas.bbox("all")))
        self._scroll_canvas.create_window((0, 0), window=self._inner, anchor="nw")
        self._scroll_canvas.configure(yscrollcommand=vsb.set,
                                      xscrollcommand=hsb.set)
        hsb.pack(side="bottom", fill="x")
        vsb.pack(side="right", fill="y")
        self._scroll_canvas.pack(fill="both", expand=True)
        self._scroll_canvas.bind_all("<MouseWheel>",
            lambda e: self._scroll_canvas.yview_scroll(
                int(-1 * (e.delta / 120)), "units"))

    # ── public API called by RunTab ───────────────────────────────────────────

    def clear(self):
        log.debug("VisualResultsTab.clear")
        for w in self._inner.winfo_children():
            w.destroy()
        self._tk_refs.clear()
        self._status_lbl.config(text="No results yet — run the pipeline first.")

    def show_results(self, sections: list):
        """
        sections: list of (section_title, list_of_(label, bgr_or_pil_image))
        """
        log.info("VisualResultsTab.show_results: %d sections", len(sections))
        self.clear()
        self._status_lbl.config(text="")
        row_idx = 0
        for section_title, items in sections:
            if not items:
                continue
            # section header
            hdr = tk.Label(self._inner, text=section_title,
                           bg="#181825", fg="#89b4fa",
                           font=("Segoe UI", 10, "bold"), anchor="w")
            hdr.grid(row=row_idx, column=0,
                     columnspan=self._COLS, sticky="w",
                     padx=8, pady=(10, 2))
            row_idx += 1
            sep = tk.Frame(self._inner, bg="#313244", height=1)
            sep.grid(row=row_idx, column=0,
                     columnspan=self._COLS, sticky="ew", padx=8, pady=(0, 6))
            row_idx += 1

            col_idx = 0
            for label, img in items:
                pil = self._to_pil(img)
                if pil is None:
                    continue
                thumb = self._make_thumb(pil)
                tk_img = ImageTk.PhotoImage(thumb)
                self._tk_refs.append(tk_img)

                cell = tk.Frame(self._inner, bg="#252535",
                                bd=1, relief="flat", pady=4, padx=4)
                cell.grid(row=row_idx, column=col_idx,
                          padx=6, pady=4, sticky="nw")

                img_lbl = tk.Label(cell, image=tk_img, bg="#252535",
                                   cursor="hand2")
                img_lbl.pack()
                # click to enlarge
                img_lbl.bind("<Button-1>",
                             lambda e, p=pil, t=label: self._enlarge(p, t))

                tk.Label(cell, text=label, bg="#252535", fg="#cdd6f4",
                         font=("Segoe UI", 8), wraplength=self._THUMB_W).pack()

                col_idx += 1
                if col_idx >= self._COLS:
                    col_idx = 0
                    row_idx += 1

            if col_idx != 0:
                row_idx += 1

    # ── helpers ───────────────────────────────────────────────────────────────

    def _to_pil(self, img) -> "Image.Image | None":
        """Accept BGR ndarray or PIL Image."""
        if img is None:
            return None
        try:
            import numpy as _np
            if isinstance(img, _np.ndarray):
                if img.size == 0:
                    return None
                if img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                return cv2_to_pil(img)
            if isinstance(img, Image.Image):
                return img
        except Exception:
            pass
        return None

    def _make_thumb(self, pil: "Image.Image") -> "Image.Image":
        pil = pil.copy()
        pil.thumbnail((self._THUMB_W, self._THUMB_H), Image.LANCZOS)
        # pad to uniform size
        canvas = Image.new("RGB", (self._THUMB_W, self._THUMB_H), (37, 37, 53))
        x = (self._THUMB_W - pil.width)  // 2
        y = (self._THUMB_H - pil.height) // 2
        canvas.paste(pil, (x, y))
        return canvas

    def _enlarge(self, pil: "Image.Image", title: str):
        """Open a resizable popup showing the full-size image."""
        popup = tk.Toplevel(self)
        popup.title(title)
        popup.configure(bg="#1e1e2e")
        sw = popup.winfo_screenwidth()
        sh = popup.winfo_screenheight()
        max_w, max_h = int(sw * 0.85), int(sh * 0.85)
        img = pil.copy()
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        lbl = tk.Label(popup, image=tk_img, bg="#1e1e2e")
        lbl.image = tk_img
        lbl.pack(padx=8, pady=8)
        tk.Button(popup, text="Close", command=popup.destroy,
                  bg="#3b82f6", fg="white", relief="flat", padx=12).pack(pady=(0, 8))


# ── Tab 6: Card Detection Lab ─────────────────────────────────────────────────

class CardDetectLabTab(tk.Frame):
    """
    Interactive lab for comparing card-outline detection methods.
    - Browse an image
    - Pick one method from a dropdown → Run Selected
    - Or click Run All → see every method side-by-side
    - Results: thumbnail with outline, score, aspect ratio, time
    - Click any thumbnail to enlarge
    """

    _THUMB_W = 260
    _THUMB_H = 180
    _COLS = 4

    def __init__(self, parent):
        log.info("CardDetectLabTab.__init__")
        super().__init__(parent, bg="#1e1e2e")
        self._img_path = tk.StringVar()
        self._method_var = tk.StringVar()
        self._template_var = tk.StringVar(value="Front")
        self._tk_refs: list = []
        self._build()

    def _build(self):
        # ── top: image path + browse ──────────────────────────────────────
        top = tk.Frame(self, bg="#1e1e2e")
        top.pack(fill="x", padx=12, pady=(12, 4))

        tk.Label(top, text="Card Detection Lab",
                 bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 13, "bold")).pack(anchor="w")

        row1 = tk.Frame(self, bg="#1e1e2e")
        row1.pack(fill="x", padx=12, pady=4)
        tk.Label(row1, text="Image:", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 10), width=8, anchor="w").pack(side="left")
        tk.Entry(row1, textvariable=self._img_path, width=60,
                 bg="#313244", fg="white",
                 insertbackground="white").pack(side="left", padx=4)
        tk.Button(row1, text="Browse…", command=self._browse,
                  bg="#7c3aed", fg="white", relief="flat", padx=8).pack(side="left")

        # ── template selector ────────────────────────────────────────────
        row_tmpl = tk.Frame(self, bg="#1e1e2e")
        row_tmpl.pack(fill="x", padx=12, pady=4)
        tk.Label(row_tmpl, text="Template:", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 10), width=8, anchor="w").pack(side="left")
        for side in ("Front", "Back"):
            tk.Radiobutton(
                row_tmpl, text=side, variable=self._template_var, value=side,
                bg="#1e1e2e", fg="#cdd6f4", selectcolor="#313244",
                activebackground="#1e1e2e", activeforeground="white",
                font=("Segoe UI", 10),
                command=self._update_template_status,
            ).pack(side="left", padx=6)
        self._tmpl_status = tk.Label(
            row_tmpl, text="", bg="#1e1e2e", fg="#6c7086",
            font=("Segoe UI", 9))
        self._tmpl_status.pack(side="left", padx=8)
        self._update_template_status()

        # ── method selector + buttons ─────────────────────────────────────
        row2 = tk.Frame(self, bg="#1e1e2e")
        row2.pack(fill="x", padx=12, pady=4)

        tk.Label(row2, text="Method:", bg="#1e1e2e", fg="#cdd6f4",
                 font=("Segoe UI", 10), width=8, anchor="w").pack(side="left")

        methods = card_detect.list_methods()
        if methods:
            self._method_var.set(methods[0])
        self._dropdown = ttk.Combobox(
            row2, textvariable=self._method_var,
            values=methods, state="readonly", width=24,
            font=("Segoe UI", 10))
        self._dropdown.pack(side="left", padx=4)

        tk.Button(row2, text="Run Selected", command=self._run_selected,
                  bg="#3b82f6", fg="white", relief="flat", padx=12,
                  font=("Segoe UI", 10, "bold")).pack(side="left", padx=8)

        tk.Button(row2, text="Run All Methods", command=self._run_all,
                  bg="#22c55e", fg="white", relief="flat", padx=12,
                  font=("Segoe UI", 10, "bold")).pack(side="left", padx=4)

        # ── status + best label ───────────────────────────────────────────
        self._status = tk.Label(self, text="", bg="#1e1e2e", fg="#f9e2af",
                                font=("Segoe UI", 10))
        self._status.pack(pady=2)

        self._best_lbl = tk.Label(self, text="", bg="#1e1e2e", fg="#a6e3a1",
                                  font=("Segoe UI", 11, "bold"))
        self._best_lbl.pack(pady=2)

        # ── scrollable results grid ───────────────────────────────────────
        outer = tk.Frame(self, bg="#181825", bd=1, relief="sunken")
        outer.pack(fill="both", expand=True, padx=12, pady=(4, 8))

        self._scroll = tk.Canvas(outer, bg="#181825", highlightthickness=0)
        vsb = ttk.Scrollbar(outer, orient="vertical", command=self._scroll.yview)
        hsb = ttk.Scrollbar(outer, orient="horizontal", command=self._scroll.xview)
        self._inner = tk.Frame(self._scroll, bg="#181825")
        self._inner.bind("<Configure>",
                         lambda e: self._scroll.configure(scrollregion=self._scroll.bbox("all")))
        self._scroll.create_window((0, 0), window=self._inner, anchor="nw")
        self._scroll.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        hsb.pack(side="bottom", fill="x")
        vsb.pack(side="right", fill="y")
        self._scroll.pack(fill="both", expand=True)
        self._scroll.bind_all("<MouseWheel>",
            lambda e: self._scroll.yview_scroll(int(-1 * (e.delta / 120)), "units"))

    # ── actions ───────────────────────────────────────────────────────────

    def _browse(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.jpg *.jpeg *.png *.bmp *.webp")])
        if path:
            self._img_path.set(path)

    def _load_image(self):
        p = self._img_path.get().strip()
        if not p:
            messagebox.showwarning("No image", "Please select an image first.")
            return None
        try:
            return tf.read_image(p)
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return None

    def _get_template_path(self):
        side = self._template_var.get()
        if side == "Front":
            return FRONT_TEMPLATE
        return BACK_TEMPLATE

    def _update_template_status(self):
        p = self._get_template_path()
        if p.exists():
            self._tmpl_status.config(text=f"({p.name} loaded)", fg="#a6e3a1")
        else:
            self._tmpl_status.config(text=f"({p.name} not found)", fg="#f38ba8")

    def _load_template(self):
        p = self._get_template_path()
        if not p.exists():
            return None
        try:
            return tf.read_image(str(p))
        except Exception:
            return None

    def _run_selected(self):
        log.info("CardDetectLabTab._run_selected")
        img = self._load_image()
        if img is None:
            return
        method = self._method_var.get()
        tmpl = self._load_template()
        self._status.config(text=f"Running {method}…", fg="#f9e2af")
        self.update_idletasks()
        threading.Thread(target=self._worker, args=(img, [method], tmpl),
                         daemon=True).start()

    def _run_all(self):
        log.info("CardDetectLabTab._run_all")
        img = self._load_image()
        if img is None:
            return
        tmpl = self._load_template()
        methods = card_detect.list_methods()
        self._status.config(text="Running all methods…", fg="#f9e2af")
        self.update_idletasks()
        threading.Thread(target=self._worker, args=(img, methods, tmpl),
                         daemon=True).start()

    def _worker(self, img, methods, template_bgr):
        log.info("CardDetectLabTab._worker: methods=%s", methods)
        results = []
        for m in methods:
            log.debug("  Running method: %s", m)
            try:
                r = card_detect.detect_card(img, m, template_bgr=template_bgr)
                results.append(r)
                log.debug("  %s -> score=%.3f", m, r.get('score', -1))
            except Exception as e:
                log.error("  %s ERROR: %s", m, e, exc_info=True)
        ranked = card_detect.rank_results(results)
        log.info("CardDetectLabTab._worker: done, %d results", len(ranked))
        self.after(0, lambda: self._show_results(ranked))

    # ── display ───────────────────────────────────────────────────────────

    def _clear(self):
        for w in self._inner.winfo_children():
            w.destroy()
        self._tk_refs.clear()
        self._best_lbl.config(text="")

    def _show_results(self, results):
        log.info("CardDetectLabTab._show_results: %d results", len(results))
        self._clear()
        if not results:
            self._status.config(text="No results.", fg="#f38ba8")
            return

        best = results[0]
        if best["contour"] is not None:
            ar_str = f"{best['aspect_ratio']:.3f}" if best['aspect_ratio'] else "?"
            self._best_lbl.config(
                text=f"Best: {best['method']}  "
                     f"(score: {best['score']:.3f},  ratio: {ar_str},  "
                     f"{best['time_ms']}ms)")
        else:
            self._best_lbl.config(text="No card detected by any method.")

        self._status.config(text=f"Done — {len(results)} method(s)", fg="#a6e3a1")

        row_idx = 0
        col_idx = 0

        for r in results:
            # Build info text
            detected = r["contour"] is not None
            ar_str = f"{r['aspect_ratio']:.3f}" if r['aspect_ratio'] else "—"
            sc_str = f"{r['score']:.3f}" if r['score'] > -1e17 else "fail"
            err_str = r.get("error") or ""
            info = (f"{r['method']}\n"
                    f"score: {sc_str}   ratio: {ar_str}\n"
                    f"time: {r['time_ms']}ms"
                    + (f"\n{err_str}" if err_str else ""))

            # Convert debug image to PIL
            debug_img = r.get("debug_img")
            if debug_img is None:
                continue
            pil = cv2_to_pil(debug_img)
            thumb = pil.copy()
            thumb.thumbnail((self._THUMB_W, self._THUMB_H), Image.LANCZOS)
            canvas_img = Image.new("RGB", (self._THUMB_W, self._THUMB_H), (37, 37, 53))
            x = (self._THUMB_W - thumb.width) // 2
            y = (self._THUMB_H - thumb.height) // 2
            canvas_img.paste(thumb, (x, y))
            tk_img = ImageTk.PhotoImage(canvas_img)
            self._tk_refs.append(tk_img)

            # Border color: green for best, red for no detection, grey otherwise
            if r is best and detected:
                border_color = "#22c55e"
            elif not detected:
                border_color = "#f38ba8"
            else:
                border_color = "#313244"

            cell = tk.Frame(self._inner, bg=border_color, bd=3, relief="solid",
                            pady=4, padx=4)
            cell.grid(row=row_idx, column=col_idx, padx=6, pady=6, sticky="nw")

            inner_cell = tk.Frame(cell, bg="#252535")
            inner_cell.pack(fill="both", expand=True)

            img_lbl = tk.Label(inner_cell, image=tk_img, bg="#252535", cursor="hand2")
            img_lbl.pack()
            img_lbl.bind("<Button-1>",
                         lambda e, p=pil, t=r['method']: self._enlarge(p, t))

            tk.Label(inner_cell, text=info, bg="#252535", fg="#cdd6f4",
                     font=("Consolas", 8), justify="left",
                     wraplength=self._THUMB_W).pack(pady=(4, 2))

            # Also show mask thumbnail if available
            mask = r.get("mask")
            if mask is not None and mask.size > 0:
                mask_pil = Image.fromarray(mask)
                mask_pil.thumbnail((self._THUMB_W // 2, self._THUMB_H // 3), Image.LANCZOS)
                mask_tk = ImageTk.PhotoImage(mask_pil)
                self._tk_refs.append(mask_tk)
                mask_lbl = tk.Label(inner_cell, image=mask_tk, bg="#252535")
                mask_lbl.pack(pady=2)
                tk.Label(inner_cell, text="mask", bg="#252535", fg="#6c7086",
                         font=("Segoe UI", 7)).pack()

            col_idx += 1
            if col_idx >= self._COLS:
                col_idx = 0
                row_idx += 1

    def _enlarge(self, pil: Image.Image, title: str):
        popup = tk.Toplevel(self)
        popup.title(title)
        popup.configure(bg="#1e1e2e")
        sw = popup.winfo_screenwidth()
        sh = popup.winfo_screenheight()
        max_w, max_h = int(sw * 0.85), int(sh * 0.85)
        img = pil.copy()
        img.thumbnail((max_w, max_h), Image.LANCZOS)
        tk_img = ImageTk.PhotoImage(img)
        lbl = tk.Label(popup, image=tk_img, bg="#1e1e2e")
        lbl.image = tk_img
        lbl.pack(padx=8, pady=8)
        tk.Button(popup, text="Close", command=popup.destroy,
                  bg="#3b82f6", fg="white", relief="flat", padx=12).pack(pady=(0, 8))


# ── Main window ───────────────────────────────────────────────────────────────

class AdminApp(tk.Tk):
    def __init__(self):
        log.info("AdminApp.__init__")
        super().__init__()
        self.title("ID Pipeline Admin")
        self.configure(bg="#1e1e2e")
        self.geometry("1100x720")
        self.resizable(True, True)

        style = ttk.Style(self)
        style.theme_use("default")
        style.configure("TNotebook", background="#1e1e2e", borderwidth=0)
        style.configure("TNotebook.Tab", background="#313244", foreground="#cdd6f4",
                         padding=[12, 6], font=("Segoe UI", 10))
        style.map("TNotebook.Tab",
                  background=[("selected", "#7c3aed")],
                  foreground=[("selected", "white")])

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=8, pady=8)

        t1 = TemplateTab(nb)
        t2 = FeaturesTab(nb)
        t3 = StagesTab(nb)
        t5 = VisualResultsTab(nb)
        t4 = RunTab(nb, t3, t5)
        t6 = CardDetectLabTab(nb)

        nb.add(t1, text="  📋 Templates & ROIs  ")
        nb.add(t2, text="  🖼 Features  ")
        nb.add(t3, text="  ⚙ Stage Toggles  ")
        nb.add(t4, text="  ▶ Run Pipeline  ")
        nb.add(t5, text="  🎨 Visual Results  ")
        nb.add(t6, text="  🔬 Detection Lab  ")

        status_bar = tk.Label(self,
            text=f"Config: {CONFIG_PATH}   |   Templates: {TEMPLATES_DIR}   |   Features: {FEATURES_DIR}",
            bg="#181825", fg="#6c7086", font=("Segoe UI", 8), anchor="w")
        status_bar.pack(fill="x", side="bottom", padx=4, pady=2)
        log.info("AdminApp.__init__ complete")

    def report_callback_exception(self, exc_type, exc_value, exc_tb):
        """Override tkinter's default exception handler to log errors instead of crashing."""
        log.error("Unhandled tkinter callback exception:",
                  exc_info=(exc_type, exc_value, exc_tb))
        msg = tb_module.format_exception(exc_type, exc_value, exc_tb)
        messagebox.showerror("Unexpected Error",
                             f"An error occurred:\n{''.join(msg[-3:])}\n\nSee admin_ui.log for details.")

    def destroy(self):
        log.info("AdminApp.destroy")
        super().destroy()


if __name__ == "__main__":
    log.info("=" * 60)
    log.info("ADMIN UI STARTING")
    log.info("=" * 60)
    try:
        TEMPLATES_DIR.mkdir(exist_ok=True)
        FEATURES_DIR.mkdir(exist_ok=True)
        (FEATURES_DIR / "front").mkdir(exist_ok=True)
        (FEATURES_DIR / "back").mkdir(exist_ok=True)
        log.info("Directories ensured")
        app = AdminApp()
        log.info("Entering mainloop")
        app.mainloop()
    except Exception as e:
        log.critical("FATAL: %s", e, exc_info=True)
        raise
    finally:
        log.info("ADMIN UI EXITED")
