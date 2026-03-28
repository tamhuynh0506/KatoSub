"""
watermark_editor.py — Interactive Watermark Region Editor

Opens a Toplevel window with:
  1. Video preview on a canvas (seekable via slider)
  2. Rectangle drawing via click-drag
  3. Per-rectangle time-range controls (start/end sliders)
  4. Confirm / Cancel buttons

Returns a list of WatermarkRegion objects to the caller.
"""

import cv2
import os
import threading
import numpy as np
import customtkinter as ctk
from tkinter import Canvas
from PIL import Image, ImageTk
from typing import List, Optional, Callable

from watermark_detector import WatermarkRegion

# ─── Color Palette (matches main.py) ─────────────────────────────────────────

COLORS = {
    "bg_dark": "#0f0f14",
    "sidebar": "#16161e",
    "card": "#1e1e2e",
    "card_hover": "#252538",
    "accent": "#7c3aed",
    "accent_hover": "#6d28d9",
    "accent_light": "#a78bfa",
    "success": "#22c55e",
    "danger": "#ef4444",
    "danger_hover": "#dc2626",
    "text_primary": "#e2e8f0",
    "text_secondary": "#94a3b8",
    "text_muted": "#64748b",
    "border": "#2d2d44",
}

FONT_FAMILY = "Segoe UI"

# Distinct colors for rectangle overlays
RECT_COLORS = [
    "#ef4444", "#f97316", "#eab308", "#22c55e",
    "#06b6d4", "#3b82f6", "#8b5cf6", "#ec4899",
]


def _format_time(seconds: float) -> str:
    """Format seconds as MM:SS."""
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


class RectangleInfo:
    """Data model for a single user-drawn rectangle + time range."""

    def __init__(self, rect_id: int, color: str):
        self.rect_id = rect_id
        self.color = color
        # Canvas coordinates (scaled to display size)
        self.x1 = self.y1 = self.x2 = self.y2 = 0
        # Time range in seconds
        self.start_sec: float = 0.0
        self.end_sec: float = 0.0
        # Reference to the canvas rectangle item id
        self.canvas_item: Optional[int] = None
        self.canvas_label: Optional[int] = None
        # UI row widgets (so we can destroy them)
        self.ui_frame: Optional[ctk.CTkFrame] = None


class WatermarkEditor(ctk.CTkToplevel):
    """
    Interactive editor for selecting watermark regions on a video.

    Usage:
        editor = WatermarkEditor(parent, video_path)
        parent.wait_window(editor)
        regions = editor.result_regions  # List[WatermarkRegion] or None
    """

    def __init__(self, parent, video_path: str):
        super().__init__(parent)

        self.video_path = video_path
        self.result_regions: Optional[List[WatermarkRegion]] = None

        # Video metadata
        cap = cv2.VideoCapture(video_path)
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self.video_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.video_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.duration_sec = self.total_frames / self.fps
        cap.release()

        # Display scaling
        self.canvas_w = 854
        self.canvas_h = int(self.canvas_w * self.video_h / self.video_w)
        self.scale_x = self.video_w / self.canvas_w
        self.scale_y = self.video_h / self.canvas_h

        # State
        self.rectangles: List[RectangleInfo] = []
        self.active_rect: Optional[RectangleInfo] = None
        self.is_drawing = False
        self.draw_start_x = 0
        self.draw_start_y = 0
        self._current_frame_idx = 0
        self._photo_image = None  # prevent GC

        # ── Window setup ──
        self.title(f"Watermark Editor — {os.path.basename(video_path)}")
        self.geometry(f"{self.canvas_w + 40}x{self.canvas_h + 380}")
        self.minsize(700, 500)
        self.configure(fg_color=COLORS["bg_dark"])
        self.transient(parent)
        self.grab_set()

        self._build_ui()
        self._seek_to_frame(0)

    # ─── UI Construction ──────────────────────────────────────────────────

    def _build_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)  # canvas
        self.grid_rowconfigure(3, weight=1)  # rectangle list

        # ── Header ──
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=16, pady=(12, 4))
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            header, text="🎯  Draw rectangles over watermark areas",
            font=(FONT_FAMILY, 14, "bold"), text_color=COLORS["text_primary"],
        ).grid(row=0, column=0, sticky="w")

        self.time_label = ctk.CTkLabel(
            header, text="00:00 / " + _format_time(self.duration_sec),
            font=(FONT_FAMILY, 12), text_color=COLORS["text_muted"],
        )
        self.time_label.grid(row=0, column=1, sticky="e")

        # ── Video Canvas ──
        canvas_frame = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=10)
        canvas_frame.grid(row=1, column=0, sticky="ew", padx=16, pady=(4, 4))

        self.canvas = Canvas(
            canvas_frame, width=self.canvas_w, height=self.canvas_h,
            bg="#000000", highlightthickness=0, cursor="crosshair",
        )
        self.canvas.pack(padx=4, pady=4)

        # Mouse bindings for rectangle drawing
        self.canvas.bind("<ButtonPress-1>", self._on_mouse_down)
        self.canvas.bind("<B1-Motion>", self._on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_mouse_up)

        # ── Timeline Slider ──
        slider_frame = ctk.CTkFrame(self, fg_color="transparent")
        slider_frame.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 4))
        slider_frame.grid_columnconfigure(0, weight=1)

        self.seek_slider = ctk.CTkSlider(
            slider_frame, from_=0, to=max(1, self.total_frames - 1),
            number_of_steps=max(1, self.total_frames - 1),
            fg_color=COLORS["card"], progress_color=COLORS["accent"],
            button_color=COLORS["accent_light"],
            button_hover_color=COLORS["accent"],
            command=self._on_seek,
        )
        self.seek_slider.set(0)
        self.seek_slider.grid(row=0, column=0, sticky="ew")

        # ── Controls Row ──
        ctrl_frame = ctk.CTkFrame(self, fg_color="transparent")
        ctrl_frame.grid(row=3, column=0, sticky="ew", padx=16, pady=(0, 4))
        ctrl_frame.grid_columnconfigure(1, weight=1)

        self.new_rect_btn = ctk.CTkButton(
            ctrl_frame, text="➕  New Rectangle", height=34,
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            font=(FONT_FAMILY, 12, "bold"),
            command=self._start_new_rectangle,
        )
        self.new_rect_btn.grid(row=0, column=0, sticky="w", padx=(0, 10))

        self.draw_hint = ctk.CTkLabel(
            ctrl_frame, text="",
            font=(FONT_FAMILY, 11), text_color=COLORS["accent_light"],
        )
        self.draw_hint.grid(row=0, column=1, sticky="w")

        # ── Rectangle List (scrollable) ──
        list_card = ctk.CTkFrame(self, fg_color=COLORS["card"], corner_radius=10)
        list_card.grid(row=4, column=0, sticky="nsew", padx=16, pady=(0, 4))
        list_card.grid_columnconfigure(0, weight=1)
        list_card.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            list_card, text="📋  Regions", font=(FONT_FAMILY, 13, "bold"),
            text_color=COLORS["text_primary"],
        ).grid(row=0, column=0, padx=12, pady=(8, 2), sticky="w")

        self.rect_list_scroll = ctk.CTkScrollableFrame(
            list_card, fg_color="transparent", height=120,
        )
        self.rect_list_scroll.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.rect_list_scroll.grid_columnconfigure(0, weight=1)

        # ── Bottom Buttons ──
        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=5, column=0, sticky="ew", padx=16, pady=(4, 14))
        btn_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(
            btn_frame, text="✅  Start Removal", height=42,
            fg_color=COLORS["success"], hover_color="#16a34a",
            font=(FONT_FAMILY, 14, "bold"),
            command=self._confirm,
        ).grid(row=0, column=0, sticky="ew", padx=(0, 8))

        ctk.CTkButton(
            btn_frame, text="Cancel", height=42, width=120,
            fg_color=COLORS["danger"], hover_color=COLORS["danger_hover"],
            font=(FONT_FAMILY, 14, "bold"),
            command=self._cancel,
        ).grid(row=0, column=1, sticky="e")

    # ─── Video Frame Display ──────────────────────────────────────────────

    def _seek_to_frame(self, frame_idx: int):
        """Read a frame from the video and display it on the canvas."""
        frame_idx = max(0, min(int(frame_idx), self.total_frames - 1))
        self._current_frame_idx = frame_idx

        cap = cv2.VideoCapture(self.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            return

        # Resize to canvas size
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (self.canvas_w, self.canvas_h))
        pil_img = Image.fromarray(frame_resized)
        self._photo_image = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("bg_image")
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo_image, tags="bg_image")

        # Re-draw all rectangle overlays on top
        self._redraw_all_rects()

        # Update time label
        current_sec = frame_idx / self.fps
        self.time_label.configure(
            text=f"{_format_time(current_sec)} / {_format_time(self.duration_sec)}"
        )

    def _on_seek(self, value):
        self._seek_to_frame(int(value))

    # ─── Rectangle Drawing ────────────────────────────────────────────────

    def _start_new_rectangle(self):
        """Enable drawing mode for a new rectangle."""
        rect_id = len(self.rectangles) + 1
        color = RECT_COLORS[(rect_id - 1) % len(RECT_COLORS)]
        ri = RectangleInfo(rect_id, color)
        # Default time range = full video
        ri.start_sec = 0.0
        ri.end_sec = self.duration_sec
        self.active_rect = ri
        self.is_drawing = False  # will become True on mouse-down

        self.draw_hint.configure(text=f"🖱  Click and drag on the video to draw R{rect_id}")
        self.new_rect_btn.configure(state="disabled")

    def _on_mouse_down(self, event):
        if self.active_rect is None:
            return
        self.is_drawing = True
        self.draw_start_x = event.x
        self.draw_start_y = event.y

        # Create temp rectangle on canvas
        r = self.active_rect
        r.canvas_item = self.canvas.create_rectangle(
            event.x, event.y, event.x, event.y,
            outline=r.color, width=2, dash=(4, 2),
        )

    def _on_mouse_drag(self, event):
        if not self.is_drawing or self.active_rect is None:
            return
        # Update the temp rectangle
        self.canvas.coords(
            self.active_rect.canvas_item,
            self.draw_start_x, self.draw_start_y, event.x, event.y,
        )

    def _on_mouse_up(self, event):
        if not self.is_drawing or self.active_rect is None:
            return
        self.is_drawing = False

        r = self.active_rect
        # Normalize coordinates
        r.x1 = min(self.draw_start_x, event.x)
        r.y1 = min(self.draw_start_y, event.y)
        r.x2 = max(self.draw_start_x, event.x)
        r.y2 = max(self.draw_start_y, event.y)

        # Minimum size check
        if abs(r.x2 - r.x1) < 10 or abs(r.y2 - r.y1) < 10:
            self.canvas.delete(r.canvas_item)
            self.draw_hint.configure(text="⚠ Rectangle too small, try again")
            self.active_rect = None
            self.new_rect_btn.configure(state="normal")
            return

        # Finalize rectangle: solid outline
        self.canvas.delete(r.canvas_item)
        r.canvas_item = self.canvas.create_rectangle(
            r.x1, r.y1, r.x2, r.y2,
            outline=r.color, width=2,
        )
        r.canvas_label = self.canvas.create_text(
            r.x1 + 4, r.y1 + 2, anchor="nw",
            text=f"R{r.rect_id}", fill=r.color,
            font=(FONT_FAMILY, 10, "bold"),
        )

        self.rectangles.append(r)
        self.active_rect = None
        self.draw_hint.configure(text="")
        self.new_rect_btn.configure(state="normal")

        # Add UI row for time range
        self._add_rect_row(r)

    # ─── Rectangle List UI ────────────────────────────────────────────────

    def _add_rect_row(self, r: RectangleInfo):
        """Add a row to the rectangle list with time range controls."""
        row = ctk.CTkFrame(self.rect_list_scroll, fg_color=COLORS["card_hover"], corner_radius=8)
        row.pack(fill="x", padx=4, pady=3)
        row.grid_columnconfigure(2, weight=1)
        row.grid_columnconfigure(5, weight=1)
        r.ui_frame = row

        # Color swatch + ID
        ctk.CTkLabel(
            row, text=f"  R{r.rect_id}", font=(FONT_FAMILY, 12, "bold"),
            text_color=r.color, width=50,
        ).grid(row=0, column=0, padx=(8, 4), pady=6, sticky="w")

        # Coordinates label
        vx1 = int(r.x1 * self.scale_x)
        vy1 = int(r.y1 * self.scale_y)
        vx2 = int(r.x2 * self.scale_x)
        vy2 = int(r.y2 * self.scale_y)
        ctk.CTkLabel(
            row, text=f"({vx1},{vy1})→({vx2},{vy2})",
            font=("Consolas", 10), text_color=COLORS["text_muted"], width=180,
        ).grid(row=0, column=1, padx=4, pady=6, sticky="w")

        # Start time
        ctk.CTkLabel(
            row, text="Start:", font=(FONT_FAMILY, 11),
            text_color=COLORS["text_secondary"],
        ).grid(row=0, column=2, padx=(8, 2), pady=6, sticky="e")

        start_slider = ctk.CTkSlider(
            row, from_=0, to=self.duration_sec, width=120,
            fg_color=COLORS["bg_dark"], progress_color=r.color,
            button_color=r.color, button_hover_color=COLORS["text_primary"],
            command=lambda val, _r=r, _lbl_ref=[None]: self._on_time_change(_r, "start", val, _lbl_ref),
        )
        start_slider.set(0)
        start_slider.grid(row=0, column=3, padx=2, pady=6, sticky="w")

        start_lbl = ctk.CTkLabel(
            row, text="00:00", font=("Consolas", 10),
            text_color=COLORS["text_secondary"], width=42,
        )
        start_lbl.grid(row=0, column=4, padx=(0, 8), pady=6, sticky="w")

        # End time
        ctk.CTkLabel(
            row, text="End:", font=(FONT_FAMILY, 11),
            text_color=COLORS["text_secondary"],
        ).grid(row=0, column=5, padx=(8, 2), pady=6, sticky="e")

        end_slider = ctk.CTkSlider(
            row, from_=0, to=self.duration_sec, width=120,
            fg_color=COLORS["bg_dark"], progress_color=r.color,
            button_color=r.color, button_hover_color=COLORS["text_primary"],
            command=lambda val, _r=r, _lbl_ref=[None]: self._on_time_change(_r, "end", val, _lbl_ref),
        )
        end_slider.set(self.duration_sec)
        end_slider.grid(row=0, column=6, padx=2, pady=6, sticky="w")

        end_lbl = ctk.CTkLabel(
            row, text=_format_time(self.duration_sec), font=("Consolas", 10),
            text_color=COLORS["text_secondary"], width=42,
        )
        end_lbl.grid(row=0, column=7, padx=(0, 8), pady=6, sticky="w")

        # Wire up label refs for dynamic updating
        start_slider.configure(
            command=lambda val, _r=r, _lbl=start_lbl: self._on_time_change(_r, "start", val, _lbl),
        )
        end_slider.configure(
            command=lambda val, _r=r, _lbl=end_lbl: self._on_time_change(_r, "end", val, _lbl),
        )

        # Delete button
        ctk.CTkButton(
            row, text="🗑", width=30, height=28,
            fg_color=COLORS["danger"], hover_color=COLORS["danger_hover"],
            font=(FONT_FAMILY, 12),
            command=lambda _r=r: self._delete_rect(_r),
        ).grid(row=0, column=8, padx=(4, 8), pady=6, sticky="e")

    def _on_time_change(self, r: RectangleInfo, which: str, value: float, label: ctk.CTkLabel):
        """Update start/end time when slider moves."""
        if which == "start":
            r.start_sec = float(value)
        else:
            r.end_sec = float(value)
        label.configure(text=_format_time(value))

    def _delete_rect(self, r: RectangleInfo):
        """Remove a rectangle from the list and canvas."""
        if r.canvas_item:
            self.canvas.delete(r.canvas_item)
        if r.canvas_label:
            self.canvas.delete(r.canvas_label)
        if r.ui_frame:
            r.ui_frame.destroy()
        if r in self.rectangles:
            self.rectangles.remove(r)

    def _redraw_all_rects(self):
        """Re-draw all rectangle overlays (called after frame seek)."""
        for r in self.rectangles:
            if r.canvas_item:
                self.canvas.delete(r.canvas_item)
            if r.canvas_label:
                self.canvas.delete(r.canvas_label)
            r.canvas_item = self.canvas.create_rectangle(
                r.x1, r.y1, r.x2, r.y2,
                outline=r.color, width=2,
            )
            r.canvas_label = self.canvas.create_text(
                r.x1 + 4, r.y1 + 2, anchor="nw",
                text=f"R{r.rect_id}", fill=r.color,
                font=(FONT_FAMILY, 10, "bold"),
            )

    # ─── Confirm / Cancel ─────────────────────────────────────────────────

    def _confirm(self):
        """Build WatermarkRegion list and close."""
        if not self.rectangles:
            self.result_regions = None
            self.grab_release()
            self.destroy()
            return

        regions = []
        for r in self.rectangles:
            # Convert canvas coords → video coords
            x = int(r.x1 * self.scale_x)
            y = int(r.y1 * self.scale_y)
            w = int((r.x2 - r.x1) * self.scale_x)
            h = int((r.y2 - r.y1) * self.scale_y)

            # Clamp
            x = max(0, x)
            y = max(0, y)
            w = min(self.video_w - x, w)
            h = min(self.video_h - y, h)

            # Convert time → frame
            start_frame = max(0, int(r.start_sec * self.fps))
            end_frame = min(self.total_frames - 1, int(r.end_sec * self.fps))

            # Ensure start < end
            if start_frame > end_frame:
                start_frame, end_frame = end_frame, start_frame

            regions.append(WatermarkRegion(x, y, w, h, start_frame, end_frame, confidence=1.0))

        self.result_regions = regions
        self.grab_release()
        self.destroy()

    def _cancel(self):
        """Close without returning regions."""
        self.result_regions = None
        self.grab_release()
        self.destroy()
