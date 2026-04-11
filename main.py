import sys
import io
import os

# Fix Windows console encoding for Unicode characters (EasyOCR progress bars etc.)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import customtkinter as ctk
from tkinter import filedialog, messagebox
import threading
import shutil
import re
from pathlib import Path
from PIL import Image, ImageDraw, ImageOps

# Granular pipeline imports for split-phase execution
from pipeline_v4 import SelectiveInpaintPipe
from srt_utils import get_stabilized_segments, frames_to_srt
from pipeline_audio import extract_audio, transcribe_audio, render_subtitles
from ai_translator import AITranslator
from pipeline_watermark import run_watermark_pipeline

# ─── Color Palette & Theme ───────────────────────────────────────────────────

COLORS = {
    "bg_dark": "#0B1426",
    "panel": "#0F1D32",
    "panel_header": "#132440",
    "card_hover": "#1A2D4A",
    "accent": "#2EA8E5",
    "accent_hover": "#1E8BC3",
    "accent_light": "#5BC0F0",
    "success": "#22C55E",
    "success_hover": "#16A34A",
    "danger": "#EF4444",
    "danger_hover": "#DC2626",
    "text_primary": "#E8EDF5",
    "text_secondary": "#94A3B8",
    "text_muted": "#5A6B82",
    "border": "#1A2D4A",
    "progress_bg": "#1A2D4A",
    "progress_fill": "#2EA8E5",
    "status_awaiting": "#F59E0B",
    "status_queue": "#6B7280",
    "status_done": "#22C55E",
}

FONT_FAMILY = "Outfit"  # Premium modern font

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# ─── Configuration & Paths ──────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
AVATAR_DIR = BASE_DIR / "data" / "avatar"
AVATAR_PATH = AVATAR_DIR / "avatar.png"
AVATAR_DIR.mkdir(parents=True, exist_ok=True)

TARGET_LANGUAGES = {
    "Vietnamese": "vi",
    "English": "en",
    "Japanese": "ja",
    "Korean": "ko",
    "Chinese (Simplified)": "zh-cn",
    "Chinese (Traditional)": "zh-tw",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Portuguese": "pt",
    "Russian": "ru",
    "Italian": "it",
    "Indonesian": "id",
    "Thai": "th",
    "Hindi": "hi",
    "Arabic": "ar",
}

RESOLUTION_PRESETS = {
    "720p 30fps": {"width": 1280, "height": 720, "fps": 30},
    "720p 60fps": {"width": 1280, "height": 720, "fps": 60},
    "1080p 30fps": {"width": 1920, "height": 1080, "fps": 30},
    "1080p 60fps": {"width": 1920, "height": 1080, "fps": 60},
    "1440p 30fps": {"width": 2560, "height": 1440, "fps": 30},
    "1440p 60fps": {"width": 2560, "height": 1440, "fps": 60},
    "Original": None,  # Keep source resolution
}

QUALITY_PRESETS = {
    "Low": {"cq": "40", "bitrate": "2M", "maxrate": "4M", "bufsize": "8M", "preset": "p4"},
    "Medium": {"cq": "35", "bitrate": "4M", "maxrate": "6M", "bufsize": "12M", "preset": "p5"},
    "High": {"cq": "28", "bitrate": "8M", "maxrate": "12M", "bufsize": "24M", "preset": "p6"},
    "Ultra": {"cq": "22", "bitrate": "15M", "maxrate": "20M", "bufsize": "40M", "preset": "p7"},
}


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("KatoSub — AI Video Toolset")
        self.geometry("1400x900")
        self.minsize(1100, 800)
        self.configure(fg_color=COLORS["bg_dark"])

        self.video_paths = []
        self.output_dir = None
        self.cancel_event = threading.Event()
        self.continue_event = threading.Event()  # Signals user approved translations
        self.is_processing = False

        # ── Translation Editor state ──
        self.translation_data = []   # List of {timestamp_start, timestamp_end, source, translated}
        self.undo_stack = []
        self.redo_stack = []
        self.glossary = {}           # {source_term: target_term}
        self.current_srt_source = ""
        self.current_srt_translated = ""
        self.translation_entry_widgets = []  # References to editable Entry widgets

        # ── Pipeline split-phase context ──
        self.pipeline_context = {}  # Stores intermediate data between phases

        # ── Layout: header + content + bottom bar ──
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=0)  # Header
        self.grid_rowconfigure(1, weight=1)  # Content
        self.grid_rowconfigure(2, weight=0)  # Bottom bar

        self._build_header_bar()
        self._build_content_area()
        self._build_bottom_bar()

    # ─── Avatar / Profile Image Helper ────────────────────────────────────

    def _get_avatar_image(self, image_path=None, size=(280, 380), corner_radius=12):
        """Creates a profile image with rounded corners and a cover effect."""
        try:
            if image_path:
                img = Image.open(image_path).convert("RGBA")
            else:
                # Default gradient background
                img = Image.new("RGBA", size)
                draw = ImageDraw.Draw(img)
                for i in range(size[1]):
                    r = int(15 + (i / size[1]) * 25)
                    g = int(25 + (i / size[1]) * 35)
                    b = int(50 + (i / size[1]) * 50)
                    draw.line([(0, i), (size[0], i)], fill=(r, g, b, 255))

            # Resize and crop to fill (cover effect)
            img_ratio = img.width / img.height
            target_ratio = size[0] / size[1]
            if img_ratio > target_ratio:
                new_width = int(target_ratio * img.height)
                offset = (img.width - new_width) // 2
                img = img.crop((offset, 0, offset + new_width, img.height))
            else:
                new_height = int(img.width / target_ratio)
                offset = (img.height - new_height) // 2
                img = img.crop((0, offset, img.width, offset + new_height))

            img = img.resize(size, Image.Resampling.LANCZOS)

            # Create rounded corner mask
            mask = Image.new("L", size, 0)
            draw = ImageDraw.Draw(mask)
            draw.rounded_rectangle((0, 0, size[0], size[1]), radius=corner_radius, fill=255)

            output = Image.new("RGBA", size, (0, 0, 0, 0))
            output.paste(img, (0, 0), mask=mask)

            return ctk.CTkImage(light_image=output, dark_image=output, size=size)
        except Exception as e:
            print(f"Error loading avatar image: {e}")
            return None

    def _change_avatar(self):
        """Opens a file dialog to change the profile image and saves it."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.png *.jpg *.jpeg *.bmp *.gif")]
        )
        if file_path:
            try:
                shutil.copy2(file_path, AVATAR_PATH)
                new_image = self._get_avatar_image(AVATAR_PATH)
                if new_image:
                    self.avatar_image = new_image
                    self.avatar_label.configure(image=self.avatar_image)
            except Exception as e:
                messagebox.showerror("Error", f"Could not save avatar: {e}")

    # ─── Header Bar ──────────────────────────────────────────────────────

    def _build_header_bar(self):
        header = ctk.CTkFrame(self, height=50, fg_color=COLORS["panel"], corner_radius=0)
        header.grid(row=0, column=0, sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        header.grid_propagate(False)

        # App title
        ctk.CTkLabel(
            header, text="  ✨ KatoSub", font=(FONT_FAMILY, 20, "bold"),
            text_color=COLORS["accent_light"], anchor="w"
        ).grid(row=0, column=0, padx=(15, 0), pady=10, sticky="w")

        # Subtitle
        ctk.CTkLabel(
            header, text="AI Video Processing Platform",
            font=(FONT_FAMILY, 11), text_color=COLORS["text_muted"], anchor="w"
        ).grid(row=0, column=1, padx=(10, 0), pady=10, sticky="w")


        # Output Path in header
        path_frame = ctk.CTkFrame(header, fg_color="transparent")
        path_frame.grid(row=0, column=2, padx=(20, 15), pady=10, sticky="e")
        
        ctk.CTkLabel(
            path_frame, text="Output Path:", font=(FONT_FAMILY, 11, "bold"),
            text_color=COLORS["text_muted"],
        ).pack(side="left", padx=(0, 8))

        self.output_entry = ctk.CTkEntry(
            path_frame, placeholder_text="Same as input video...", width=250, height=28,
            fg_color=COLORS["bg_dark"], border_color=COLORS["border"],
            text_color=COLORS["text_primary"], font=(FONT_FAMILY, 11),
        )
        self.output_entry.pack(side="left", padx=(0, 8))

        ctk.CTkButton(
            path_frame, text="Browse...", width=65, height=28,
            fg_color=COLORS["card_hover"], hover_color=COLORS["border"],
            font=(FONT_FAMILY, 11), command=self._browse_output,
        ).pack(side="left")
    # ─── Content Area (main grid) ─────────────────────────────────────────

    def _build_content_area(self):
        content = ctk.CTkFrame(self, fg_color="transparent")
        content.grid(row=1, column=0, sticky="nsew", padx=8, pady=(8, 4))

        # 3 columns: avatar / settings | queue / pipeline log | editor / trans log
        content.grid_columnconfigure(0, weight=0, minsize=320)  # Avatar & Settings col
        content.grid_columnconfigure(1, weight=1, minsize=280)  # Queue & Pipeline col
        content.grid_columnconfigure(2, weight=3, minsize=500)  # Editor & Translation log col

        # 2 rows: top panels | log/settings panels
        content.grid_rowconfigure(0, weight=4)  # Top panels (taller)
        content.grid_rowconfigure(1, weight=3)  # Bottom panels

        self._build_avatar_panel(content)
        self._build_queue_panel(content)
        self._build_translation_editor(content)
        
        self._build_settings_panel(content)
        self._build_log_panels(content)

    # ─── Avatar Panel (far left) ──────────────────────────────────────────

    def _build_avatar_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color=COLORS["panel"], corner_radius=12)
        panel.grid(row=0, column=0, sticky="nsew", padx=(0, 6), pady=(0, 6))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(0, weight=1)

        default_img = AVATAR_PATH if AVATAR_PATH.exists() else None
        self.avatar_image = self._get_avatar_image(default_img)
        self.avatar_label = ctk.CTkLabel(
            panel, text="" if self.avatar_image else "Click to\nset image",
            image=self.avatar_image,
            font=(FONT_FAMILY, 13, "bold"),
            text_color=COLORS["text_muted"],
            cursor="hand2",
        )
        self.avatar_label.grid(row=0, column=0, padx=6, pady=6, sticky="nsew")
        self.avatar_label.bind("<Button-1>", lambda e: self._change_avatar())

    # ─── Queue Panel (center-left) ────────────────────────────────────────

    def _build_queue_panel(self, parent):
        panel = ctk.CTkFrame(parent, fg_color=COLORS["panel"], corner_radius=12)
        panel.grid(row=0, column=1, sticky="nsew", padx=(0, 6), pady=(0, 6))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        # Header
        header = ctk.CTkFrame(panel, fg_color=COLORS["panel_header"], corner_radius=8)
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header, text="Compact Queue & Add/Edit",
            font=(FONT_FAMILY, 13, "bold"), text_color=COLORS["text_primary"],
        ).grid(row=0, column=0, padx=12, pady=8, sticky="w")

        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.grid(row=0, column=1, padx=(0, 8), pady=6, sticky="e")

        ctk.CTkButton(
            btn_frame, text="+ Add Videos", width=90, height=26,
            fg_color=COLORS["success"], hover_color=COLORS["success_hover"],
            font=(FONT_FAMILY, 10, "bold"), command=self._add_videos,
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            btn_frame, text="Remove", width=60, height=26,
            fg_color=COLORS["danger"], hover_color=COLORS["danger_hover"],
            font=(FONT_FAMILY, 10), command=self._remove_selected,
        ).pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            btn_frame, text="Clear", width=50, height=26,
            fg_color=COLORS["card_hover"], hover_color=COLORS["border"],
            font=(FONT_FAMILY, 10), command=self._clear_queue,
        ).pack(side="left")

        # Table header row
        table_header = ctk.CTkFrame(panel, fg_color=COLORS["panel_header"], corner_radius=6, height=30)
        table_header.grid(row=1, column=0, sticky="new", padx=8, pady=(2, 0))
        table_header.grid_columnconfigure(0, weight=3)
        table_header.grid_columnconfigure(1, weight=1)
        table_header.grid_propagate(False)

        ctk.CTkLabel(
            table_header, text="File", font=(FONT_FAMILY, 11, "bold"),
            text_color=COLORS["text_muted"],
        ).grid(row=0, column=0, padx=12, pady=4, sticky="w")

        ctk.CTkLabel(
            table_header, text="Status", font=(FONT_FAMILY, 11, "bold"),
            text_color=COLORS["text_muted"],
        ).grid(row=0, column=1, padx=12, pady=4, sticky="e")

        # Scrollable file list
        self.queue_scroll = ctk.CTkScrollableFrame(
            panel, fg_color=COLORS["bg_dark"], corner_radius=8,
            scrollbar_button_color=COLORS["border"],
            scrollbar_button_hover_color=COLORS["accent"],
        )
        self.queue_scroll.grid(row=1, column=0, sticky="nsew", padx=8, pady=(32, 8))
        self.queue_scroll.grid_columnconfigure(0, weight=3)
        self.queue_scroll.grid_columnconfigure(1, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        self._refresh_file_list()

    # ─── Interactive Translation Editor (MAIN PANEL) ───────────────────────

    def _build_translation_editor(self, parent):
        panel = ctk.CTkFrame(parent, fg_color=COLORS["panel"], corner_radius=12)
        panel.grid(row=0, column=2, sticky="nsew", pady=(0, 6))
        panel.grid_columnconfigure(0, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        # Header bar
        header = ctk.CTkFrame(panel, fg_color=COLORS["panel_header"], corner_radius=8)
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        header.grid_columnconfigure(0, weight=0)
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            header, text="Interactive Translation Editor",
            font=(FONT_FAMILY, 13, "bold"), text_color=COLORS["text_primary"],
        ).grid(row=0, column=0, padx=12, pady=8, sticky="w")

        # Editor toolbar
        toolbar = ctk.CTkFrame(header, fg_color="transparent")
        toolbar.grid(row=0, column=1, padx=(0, 8), pady=6, sticky="e")

        ctk.CTkButton(
            toolbar, text="↶", width=30, height=26,
            fg_color=COLORS["card_hover"], hover_color=COLORS["border"],
            font=(FONT_FAMILY, 14), command=self._undo_translation,
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            toolbar, text="↷", width=30, height=26,
            fg_color=COLORS["card_hover"], hover_color=COLORS["border"],
            font=(FONT_FAMILY, 14), command=self._redo_translation,
        ).pack(side="left", padx=2)

        ctk.CTkButton(
            toolbar, text="📖 Glossary", width=90, height=26,
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            font=(FONT_FAMILY, 11, "bold"), command=self._open_glossary,
        ).pack(side="left", padx=(8, 0))

        # Continue Inpainting button — shown after OCR+Translation completes
        self.continue_btn = ctk.CTkButton(
            toolbar, text="▶ Continue Inpainting", width=160, height=28,
            fg_color=COLORS["success"], hover_color=COLORS["success_hover"],
            font=(FONT_FAMILY, 11, "bold"), command=self._on_continue_clicked,
        )
        # Hidden by default, shown when pipeline is waiting for user review
        self.continue_btn.pack_forget()

        # Column headers for source / translated
        col_header = ctk.CTkFrame(panel, fg_color=COLORS["panel_header"], corner_radius=6, height=30)
        col_header.grid(row=1, column=0, sticky="new", padx=8, pady=(2, 0))
        col_header.grid_columnconfigure(0, weight=0, minsize=55)
        col_header.grid_columnconfigure(1, weight=1)
        col_header.grid_columnconfigure(2, weight=0, minsize=55)
        col_header.grid_columnconfigure(3, weight=1)
        col_header.grid_propagate(False)

        ctk.CTkLabel(
            col_header, text="Time", font=(FONT_FAMILY, 10, "bold"),
            text_color=COLORS["text_muted"],
        ).grid(row=0, column=0, padx=(12, 4), pady=4, sticky="w")

        ctk.CTkLabel(
            col_header, text="Source Text", font=(FONT_FAMILY, 10, "bold"),
            text_color=COLORS["text_muted"],
        ).grid(row=0, column=1, padx=4, pady=4, sticky="w")

        ctk.CTkLabel(
            col_header, text="Time", font=(FONT_FAMILY, 10, "bold"),
            text_color=COLORS["text_muted"],
        ).grid(row=0, column=2, padx=4, pady=4, sticky="w")

        ctk.CTkLabel(
            col_header, text="Translated Text", font=(FONT_FAMILY, 10, "bold"),
            text_color=COLORS["text_muted"],
        ).grid(row=0, column=3, padx=4, pady=4, sticky="w")

        # Scrollable editor rows
        self.editor_scroll = ctk.CTkScrollableFrame(
            panel, fg_color=COLORS["bg_dark"], corner_radius=8,
            scrollbar_button_color=COLORS["border"],
            scrollbar_button_hover_color=COLORS["accent"],
        )
        self.editor_scroll.grid(row=1, column=0, sticky="nsew", padx=8, pady=(32, 8))
        self.editor_scroll.grid_columnconfigure(0, weight=0, minsize=55)
        self.editor_scroll.grid_columnconfigure(1, weight=1)
        self.editor_scroll.grid_columnconfigure(2, weight=0, minsize=55)
        self.editor_scroll.grid_columnconfigure(3, weight=1)
        panel.grid_rowconfigure(1, weight=1)

        # Show placeholder
        self._render_editor_placeholder()

    def _render_editor_placeholder(self):
        """Show empty state in the translation editor."""
        for w in self.editor_scroll.winfo_children():
            w.destroy()
        self.translation_entry_widgets.clear()

        ctk.CTkLabel(
            self.editor_scroll, text="No translation data loaded.\nProcess a video to see subtitles here.",
            font=(FONT_FAMILY, 12), text_color=COLORS["text_muted"],
        ).grid(row=0, column=0, columnspan=4, pady=40, sticky="ew")

    def _load_translation_data(self, srt_source, srt_translated):
        """Parse two SRT strings into paired entries and render in editor."""
        self.current_srt_source = srt_source
        self.current_srt_translated = srt_translated
        self.translation_data.clear()
        self.undo_stack.clear()
        self.redo_stack.clear()

        source_blocks = self._parse_srt(srt_source)
        trans_blocks = self._parse_srt(srt_translated)

        # Pair them up (by index)
        max_len = max(len(source_blocks), len(trans_blocks))
        for i in range(max_len):
            src = source_blocks[i] if i < len(source_blocks) else {"time_start": "", "time_end": "", "text": ""}
            trn = trans_blocks[i] if i < len(trans_blocks) else {"time_start": "", "time_end": "", "text": ""}
            self.translation_data.append({
                "timestamp_start": src.get("time_start", ""),
                "timestamp_end": src.get("time_end", ""),
                "source": src.get("text", ""),
                "translated": trn.get("text", ""),
            })

        self._render_translation_rows()
        self._translation_log("Translation editor loaded with {} entries.".format(len(self.translation_data)))

    def _parse_srt(self, srt_content):
        """Parse SRT content into a list of dicts with time_start, time_end, text."""
        blocks = []
        if not srt_content or not srt_content.strip():
            return blocks

        raw_blocks = re.split(r'\n\n+', srt_content.strip())
        for block in raw_blocks:
            lines = block.strip().split('\n')
            if len(lines) >= 3:
                time_line = lines[1]
                text = "\n".join(lines[2:]).strip()
                parts = time_line.split(" --> ")
                time_start = parts[0].strip()[:5] if parts else ""  # HH:MM
                time_end = parts[1].strip()[:5] if len(parts) > 1 else ""
                blocks.append({"time_start": time_start, "time_end": time_end, "text": text})
            elif len(lines) >= 2:
                blocks.append({"time_start": "", "time_end": "", "text": lines[-1].strip()})

        return blocks

    def _render_translation_rows(self):
        """Build the scrollable side-by-side editor rows."""
        for w in self.editor_scroll.winfo_children():
            w.destroy()
        self.translation_entry_widgets.clear()

        for i, entry in enumerate(self.translation_data):
            row_bg = COLORS["bg_dark"] if i % 2 == 0 else COLORS["panel"]

            # Source timestamp
            ctk.CTkLabel(
                self.editor_scroll, text=entry["timestamp_start"],
                font=("Consolas", 10), text_color=COLORS["text_muted"],
                width=55,
            ).grid(row=i, column=0, padx=(8, 4), pady=2, sticky="w")

            # Source text (read-only)
            src_entry = ctk.CTkEntry(
                self.editor_scroll, font=(FONT_FAMILY, 11),
                fg_color=row_bg, text_color=COLORS["text_secondary"],
                border_width=0, state="disabled",
            )
            src_entry.grid(row=i, column=1, padx=2, pady=2, sticky="ew")
            src_entry.configure(state="normal")
            src_entry.insert(0, entry["source"])
            src_entry.configure(state="disabled")

            # Translated timestamp
            ctk.CTkLabel(
                self.editor_scroll, text=entry["timestamp_start"],
                font=("Consolas", 10), text_color=COLORS["text_muted"],
                width=55,
            ).grid(row=i, column=2, padx=4, pady=2, sticky="w")

            # Translated text (editable)
            trans_entry = ctk.CTkEntry(
                self.editor_scroll, font=(FONT_FAMILY, 11),
                fg_color=row_bg, text_color=COLORS["text_primary"],
                border_width=1, border_color=COLORS["border"],
            )
            trans_entry.grid(row=i, column=3, padx=(2, 8), pady=2, sticky="ew")
            trans_entry.insert(0, entry["translated"])

            # Bind edit tracking
            idx = i
            trans_entry.bind("<FocusOut>", lambda e, _i=idx: self._on_translation_edit(_i, e.widget.get()))

            self.translation_entry_widgets.append(trans_entry)

    def _on_translation_edit(self, index, new_text):
        """Track edits for undo support."""
        if index < len(self.translation_data):
            old_text = self.translation_data[index]["translated"]
            if old_text != new_text:
                self.undo_stack.append({"index": index, "old": old_text, "new": new_text})
                self.redo_stack.clear()
                self.translation_data[index]["translated"] = new_text
                self._translation_log(f"Edited line {index + 1}")

    def _undo_translation(self):
        """Undo last translation edit."""
        if not self.undo_stack:
            return
        action = self.undo_stack.pop()
        idx = action["index"]
        self.translation_data[idx]["translated"] = action["old"]
        self.redo_stack.append(action)

        # Update widget
        if idx < len(self.translation_entry_widgets):
            w = self.translation_entry_widgets[idx]
            w.delete(0, "end")
            w.insert(0, action["old"])
        self._translation_log(f"Undo: line {idx + 1}")

    def _redo_translation(self):
        """Redo last undone translation edit."""
        if not self.redo_stack:
            return
        action = self.redo_stack.pop()
        idx = action["index"]
        self.translation_data[idx]["translated"] = action["new"]
        self.undo_stack.append(action)

        if idx < len(self.translation_entry_widgets):
            w = self.translation_entry_widgets[idx]
            w.delete(0, "end")
            w.insert(0, action["new"])
        self._translation_log(f"Redo: line {idx + 1}")

    def _open_glossary(self):
        """Open glossary management popup."""
        popup = ctk.CTkToplevel(self)
        popup.title("Glossary — Term Consistency")
        popup.geometry("500x450")
        popup.configure(fg_color=COLORS["bg_dark"])
        popup.transient(self)
        popup.grab_set()

        ctk.CTkLabel(
            popup, text="📖 Glossary",
            font=(FONT_FAMILY, 18, "bold"), text_color=COLORS["accent_light"],
        ).pack(padx=20, pady=(15, 5))

        ctk.CTkLabel(
            popup, text="Define term mappings for consistent translations.",
            font=(FONT_FAMILY, 11), text_color=COLORS["text_muted"],
        ).pack(padx=20, pady=(0, 10))

        # Add term row
        add_frame = ctk.CTkFrame(popup, fg_color=COLORS["panel"], corner_radius=8)
        add_frame.pack(fill="x", padx=15, pady=(0, 10))

        ctk.CTkLabel(add_frame, text="Source:", font=(FONT_FAMILY, 11),
                     text_color=COLORS["text_secondary"]).grid(row=0, column=0, padx=(10, 4), pady=8)
        src_entry = ctk.CTkEntry(add_frame, width=150, fg_color=COLORS["bg_dark"],
                                 border_color=COLORS["border"], text_color=COLORS["text_primary"])
        src_entry.grid(row=0, column=1, padx=4, pady=8)

        ctk.CTkLabel(add_frame, text="→", font=(FONT_FAMILY, 14),
                     text_color=COLORS["text_muted"]).grid(row=0, column=2, padx=4)

        ctk.CTkLabel(add_frame, text="Target:", font=(FONT_FAMILY, 11),
                     text_color=COLORS["text_secondary"]).grid(row=0, column=3, padx=4, pady=8)
        tgt_entry = ctk.CTkEntry(add_frame, width=150, fg_color=COLORS["bg_dark"],
                                 border_color=COLORS["border"], text_color=COLORS["text_primary"])
        tgt_entry.grid(row=0, column=4, padx=4, pady=8)

        def _add_term():
            s = src_entry.get().strip()
            t = tgt_entry.get().strip()
            if s and t:
                self.glossary[s] = t
                src_entry.delete(0, "end")
                tgt_entry.delete(0, "end")
                _refresh_glossary_list()
                self._translation_log(f"Glossary: added '{s}' → '{t}'")

        ctk.CTkButton(
            add_frame, text="+ Add", width=60, height=28,
            fg_color=COLORS["success"], hover_color=COLORS["success_hover"],
            font=(FONT_FAMILY, 11, "bold"), command=_add_term,
        ).grid(row=0, column=5, padx=(8, 10), pady=8)

        # Glossary list
        glossary_scroll = ctk.CTkScrollableFrame(
            popup, fg_color=COLORS["panel"], corner_radius=8,
            scrollbar_button_color=COLORS["border"],
        )
        glossary_scroll.pack(fill="both", expand=True, padx=15, pady=(0, 10))
        glossary_scroll.grid_columnconfigure(0, weight=1)
        glossary_scroll.grid_columnconfigure(1, weight=0)
        glossary_scroll.grid_columnconfigure(2, weight=1)

        def _refresh_glossary_list():
            for w in glossary_scroll.winfo_children():
                w.destroy()
            for i, (s, t) in enumerate(self.glossary.items()):
                ctk.CTkLabel(glossary_scroll, text=s, font=(FONT_FAMILY, 11),
                             text_color=COLORS["text_primary"]).grid(row=i, column=0, padx=10, pady=3, sticky="w")
                ctk.CTkLabel(glossary_scroll, text="→", font=(FONT_FAMILY, 12),
                             text_color=COLORS["text_muted"]).grid(row=i, column=1, padx=4, pady=3)
                ctk.CTkLabel(glossary_scroll, text=t, font=(FONT_FAMILY, 11),
                             text_color=COLORS["accent_light"]).grid(row=i, column=2, padx=10, pady=3, sticky="w")

                def _remove(key=s):
                    del self.glossary[key]
                    _refresh_glossary_list()

                ctk.CTkButton(glossary_scroll, text="✕", width=28, height=24,
                              fg_color=COLORS["danger"], hover_color=COLORS["danger_hover"],
                              font=(FONT_FAMILY, 10), command=_remove).grid(row=i, column=3, padx=4, pady=3)

        _refresh_glossary_list()

        # Export button
        ctk.CTkButton(
            popup, text="Apply Glossary to Translations", height=36,
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            font=(FONT_FAMILY, 12, "bold"), command=lambda: self._apply_glossary(popup),
        ).pack(fill="x", padx=15, pady=(0, 15))

    def _apply_glossary(self, popup=None):
        """Apply glossary replacements to all translated text."""
        if not self.glossary or not self.translation_data:
            return
        count = 0
        for i, entry in enumerate(self.translation_data):
            original = entry["translated"]
            modified = original
            for src_term, tgt_term in self.glossary.items():
                modified = modified.replace(src_term, tgt_term)
            if modified != original:
                self.undo_stack.append({"index": i, "old": original, "new": modified})
                entry["translated"] = modified
                if i < len(self.translation_entry_widgets):
                    w = self.translation_entry_widgets[i]
                    w.delete(0, "end")
                    w.insert(0, modified)
                count += 1

        self._translation_log(f"Glossary applied: {count} entries modified")
        if popup:
            popup.destroy()

    def _export_edited_srt(self):
        """Export the edited translations back to an SRT file."""
        if not self.translation_data:
            messagebox.showinfo("Info", "No translation data to export.")
            return

        file_path = filedialog.asksaveasfilename(
            defaultextension=".srt",
            filetypes=[("SRT Files", "*.srt"), ("All Files", "*.*")],
            title="Export Edited SRT"
        )
        if not file_path:
            return

        # Rebuild SRT from translation_data
        srt_lines = []
        for i, entry in enumerate(self.translation_data):
            srt_lines.append(str(i + 1))
            # Reconstruct full timestamp from source SRT
            source_blocks = self._parse_srt(self.current_srt_source)
            if i < len(source_blocks):
                # Use original timestamp line
                raw_blocks = re.split(r'\n\n+', self.current_srt_source.strip())
                if i < len(raw_blocks):
                    lines = raw_blocks[i].strip().split('\n')
                    if len(lines) >= 2:
                        srt_lines.append(lines[1])  # Original timestamp line
                    else:
                        srt_lines.append(f"{entry['timestamp_start']}:00,000 --> {entry['timestamp_end']}:00,000")
                else:
                    srt_lines.append(f"{entry['timestamp_start']}:00,000 --> {entry['timestamp_end']}:00,000")
            else:
                srt_lines.append(f"{entry['timestamp_start']}:00,000 --> {entry['timestamp_end']}:00,000")
            srt_lines.append(entry["translated"])
            srt_lines.append("")

        with open(file_path, "w", encoding="utf-8-sig") as f:
            f.write("\n".join(srt_lines))

        self._translation_log(f"Exported edited SRT to {os.path.basename(file_path)}")

    # ─── Settings Panel ───────────────────────────────────────────────────

    def _build_settings_panel(self, parent):
        # Positioned in row 1, col 0 (matches Avatar width)
        panel = ctk.CTkFrame(parent, fg_color=COLORS["panel"], corner_radius=12)
        panel.grid(row=1, column=0, sticky="nsew", padx=(0, 6), pady=(0, 0))
        panel.grid_columnconfigure(0, weight=1)

        # Header
        header = ctk.CTkFrame(panel, fg_color=COLORS["panel_header"], corner_radius=8)
        header.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 6))
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header, text="Project Settings",
            font=(FONT_FAMILY, 13, "bold"), text_color=COLORS["text_primary"],
        ).grid(row=0, column=0, padx=12, pady=8, sticky="w")

        # Container for horizontal grid items
        settings_container = ctk.CTkScrollableFrame(
            panel, fg_color="transparent", corner_radius=0,
            scrollbar_button_color=COLORS["border"],
            scrollbar_button_hover_color=COLORS["accent"],
        )
        settings_container.grid(row=1, column=0, sticky="nsew", padx=4, pady=0)
        panel.grid_rowconfigure(1, weight=1)
        settings_container.grid_columnconfigure(0, weight=1)
        settings_container.grid_columnconfigure(1, weight=1)

        # Row 0: Pipeline Type
        type_frame = ctk.CTkFrame(settings_container, fg_color="transparent")
        type_frame.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 8), padx=4)
        type_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            type_frame, text="Type", font=(FONT_FAMILY, 10, "bold"),
            text_color=COLORS["text_muted"],
        ).grid(row=0, column=0, padx=4, pady=(0, 2), sticky="w")

        self.pipeline_mode_var = ctk.StringVar(value="Hardcoded Subs (OCR)")
        ctk.CTkOptionMenu(
            type_frame, variable=self.pipeline_mode_var,
            values=["Hardcoded Subs (OCR)", "Audio Only (Whisper)", "Replace Subs (Full)", "Watermark Removal"],
            fg_color=COLORS["bg_dark"], button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
            dropdown_fg_color=COLORS["panel"],
            font=(FONT_FAMILY, 11),
            command=self._on_pipeline_mode_change,
        ).grid(row=1, column=0, sticky="ew", padx=4)

        # Row 1: Left = Resolution Preset
        res_frame = ctk.CTkFrame(settings_container, fg_color="transparent")
        res_frame.grid(row=1, column=0, sticky="ew", pady=(0, 8), padx=4)
        res_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            res_frame, text="Resolution Preset", font=(FONT_FAMILY, 10, "bold"),
            text_color=COLORS["text_muted"],
        ).grid(row=0, column=0, padx=4, pady=(0, 2), sticky="w")

        self.resolution_var = ctk.StringVar(value="1080p 60fps")
        ctk.CTkOptionMenu(
            res_frame, variable=self.resolution_var,
            values=list(RESOLUTION_PRESETS.keys()),
            fg_color=COLORS["bg_dark"], button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
            dropdown_fg_color=COLORS["panel"],
            font=(FONT_FAMILY, 11),
        ).grid(row=1, column=0, sticky="ew", padx=4)

        # Row 1: Right = Output Quality
        quality_frame = ctk.CTkFrame(settings_container, fg_color="transparent")
        quality_frame.grid(row=1, column=1, sticky="ew", pady=(0, 8), padx=4)
        quality_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            quality_frame, text="Output Quality", font=(FONT_FAMILY, 10, "bold"),
            text_color=COLORS["text_muted"],
        ).grid(row=0, column=0, padx=4, pady=(0, 2), sticky="w")

        self.quality_var = ctk.StringVar(value="High")
        ctk.CTkOptionMenu(
            quality_frame, variable=self.quality_var,
            values=list(QUALITY_PRESETS.keys()),
            fg_color=COLORS["bg_dark"], button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
            dropdown_fg_color=COLORS["panel"],
            font=(FONT_FAMILY, 11),
        ).grid(row=1, column=0, sticky="ew", padx=4)

        # Row 2: Left = Translator
        translator_frame = ctk.CTkFrame(settings_container, fg_color="transparent")
        translator_frame.grid(row=2, column=0, sticky="ew", pady=(0, 8), padx=4)
        translator_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            translator_frame, text="Translator", font=(FONT_FAMILY, 10, "bold"),
            text_color=COLORS["text_muted"],
        ).grid(row=0, column=0, padx=4, pady=(0, 2), sticky="w")

        self.translator_var = ctk.StringVar(value="Google Translate")
        ctk.CTkOptionMenu(
            translator_frame, variable=self.translator_var,
            values=["Google Translate", "ChatGPT", "Ollama (gemma3:12b)", "Ollama (gemma3:27b)"],
            fg_color=COLORS["bg_dark"], button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
            dropdown_fg_color=COLORS["panel"],
            font=(FONT_FAMILY, 11),
        ).grid(row=1, column=0, sticky="ew", padx=4)

        # Row 2: Right = Target Language
        lang_frame = ctk.CTkFrame(settings_container, fg_color="transparent")
        lang_frame.grid(row=2, column=1, sticky="ew", pady=(0, 8), padx=4)
        lang_frame.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            lang_frame, text="Target Language", font=(FONT_FAMILY, 10, "bold"),
            text_color=COLORS["text_muted"],
        ).grid(row=0, column=0, padx=4, pady=(0, 2), sticky="w")

        self.target_lang_var = ctk.StringVar(value="Vietnamese")
        ctk.CTkOptionMenu(
            lang_frame, variable=self.target_lang_var,
            values=list(TARGET_LANGUAGES.keys()),
            fg_color=COLORS["bg_dark"], button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
            dropdown_fg_color=COLORS["panel"],
            font=(FONT_FAMILY, 11),
        ).grid(row=1, column=0, sticky="ew", padx=4)

        # Row 3: Left = Whisper model (shown conditionally)
        self.whisper_frame = ctk.CTkFrame(settings_container, fg_color="transparent")
        self.whisper_frame.grid(row=3, column=0, sticky="ew", pady=(0, 8), padx=4)
        self.whisper_frame.grid_columnconfigure(0, weight=1)

        self.whisper_label = ctk.CTkLabel(
            self.whisper_frame, text="Whisper Model", font=(FONT_FAMILY, 10, "bold"),
            text_color=COLORS["text_muted"],
        )
        self.whisper_label.grid(row=0, column=0, padx=4, pady=(0, 2), sticky="w")

        self.whisper_model_var = ctk.StringVar(value="base")
        self.whisper_menu = ctk.CTkOptionMenu(
            self.whisper_frame, variable=self.whisper_model_var,
            values=["tiny", "base", "small", "medium"],
            fg_color=COLORS["bg_dark"], button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
            dropdown_fg_color=COLORS["panel"],
            font=(FONT_FAMILY, 11),
        )
        self.whisper_menu.grid(row=1, column=0, sticky="ew", padx=4)
        self.whisper_frame.grid_remove() # hide by default

    def _on_pipeline_mode_change(self, value):
        """Show/hide Whisper model selector based on pipeline mode."""
        self.whisper_frame.grid_remove()

        if value == "Audio Only (Whisper)":
            self.whisper_frame.grid()

    # ─── Log Panels (dual split) ─────────────────────────────────────────

    def _build_log_panels(self, parent):
        # Left: Pipeline Processing Log (Maps to Queue in Col 1)
        left_log = ctk.CTkFrame(parent, fg_color=COLORS["panel"], corner_radius=12)
        left_log.grid(row=1, column=1, sticky="nsew", padx=(0, 4), pady=(0, 0))
        left_log.grid_columnconfigure(0, weight=1)
        left_log.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            left_log, text="📋  Processing Log",
            font=(FONT_FAMILY, 12, "bold"), text_color=COLORS["text_primary"],
        ).grid(row=0, column=0, padx=12, pady=(8, 4), sticky="w")

        self.log_console = ctk.CTkTextbox(
            left_log, font=("Consolas", 10),
            fg_color=COLORS["bg_dark"], text_color=COLORS["text_muted"],
            corner_radius=8, border_width=1, border_color=COLORS["border"],
        )
        self.log_console.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.log_console.configure(state="disabled")

        # Right: Translation Log (Maps to Editor in Col 2)
        right_log = ctk.CTkFrame(parent, fg_color=COLORS["panel"], corner_radius=12)
        right_log.grid(row=1, column=2, sticky="nsew", padx=(4, 0), pady=(0, 0))
        right_log.grid_columnconfigure(0, weight=1)
        right_log.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            right_log, text="📝  Translation Log",
            font=(FONT_FAMILY, 12, "bold"), text_color=COLORS["text_primary"],
        ).grid(row=0, column=0, padx=12, pady=(8, 4), sticky="w")

        self.translation_log_console = ctk.CTkTextbox(
            right_log, font=("Consolas", 10),
            fg_color=COLORS["bg_dark"], text_color=COLORS["text_muted"],
            corner_radius=8, border_width=1, border_color=COLORS["border"],
        )
        self.translation_log_console.grid(row=1, column=0, sticky="nsew", padx=8, pady=(0, 8))
        self.translation_log_console.configure(state="disabled")

    # ─── Bottom Bar (Unified Processing) ──────────────────────────────────

    def _build_bottom_bar(self):
        bar = ctk.CTkFrame(self, height=55, fg_color=COLORS["panel"], corner_radius=0)
        bar.grid(row=2, column=0, sticky="ew")
        bar.grid_propagate(False)
        bar.grid_columnconfigure(2, weight=1)  # Progress bar stretches

        # Unified Processing label
        ctk.CTkLabel(
            bar, text="Unified Processing",
            font=(FONT_FAMILY, 12, "bold"), text_color=COLORS["text_primary"],
        ).grid(row=0, column=0, padx=(15, 10), pady=12, sticky="w")

        # Status / project name
        self.status_label = ctk.CTkLabel(
            bar, text="Awaiting Rendering",
            font=(FONT_FAMILY, 11), text_color=COLORS["text_muted"],
        )
        self.status_label.grid(row=0, column=1, padx=(0, 15), pady=12, sticky="w")

        # Progress bar area
        progress_frame = ctk.CTkFrame(bar, fg_color="transparent")
        progress_frame.grid(row=0, column=2, sticky="ew", padx=10, pady=12)
        progress_frame.grid_columnconfigure(0, weight=0)
        progress_frame.grid_columnconfigure(1, weight=1)
        progress_frame.grid_columnconfigure(2, weight=0)

        ctk.CTkLabel(
            progress_frame, text="Progress Bar",
            font=(FONT_FAMILY, 10), text_color=COLORS["text_muted"],
        ).grid(row=0, column=0, padx=(0, 8), sticky="w")

        self.progress_bar = ctk.CTkProgressBar(
            progress_frame, height=12, corner_radius=6,
            fg_color=COLORS["progress_bg"], progress_color=COLORS["progress_fill"],
        )
        self.progress_bar.grid(row=0, column=1, sticky="ew", padx=4)
        self.progress_bar.set(0)

        self.progress_pct_label = ctk.CTkLabel(
            progress_frame, text="0%",
            font=(FONT_FAMILY, 10, "bold"), text_color=COLORS["accent_light"],
        )
        self.progress_pct_label.grid(row=0, column=2, padx=(6, 0), sticky="e")

        # Remaining / ETA
        self.eta_label = ctk.CTkLabel(
            bar, text="Remaining: ---",
            font=(FONT_FAMILY, 10), text_color=COLORS["text_muted"],
        )
        self.eta_label.grid(row=0, column=3, padx=10, pady=12, sticky="e")

        # Cancel button
        self.cancel_btn = ctk.CTkButton(
            bar, text="Cancel", width=80, height=34,
            fg_color=COLORS["card_hover"], hover_color=COLORS["border"],
            font=(FONT_FAMILY, 12, "bold"), command=self._cancel_processing,
            state="disabled",
        )
        self.cancel_btn.grid(row=0, column=4, padx=(5, 4), pady=10)

        # Process button
        self.start_btn = ctk.CTkButton(
            bar, text="Process", width=100, height=34,
            fg_color=COLORS["success"], hover_color=COLORS["success_hover"],
            font=(FONT_FAMILY, 12, "bold"), command=self._start_processing,
        )
        self.start_btn.grid(row=0, column=5, padx=(4, 15), pady=10)

    # ─── File Management ──────────────────────────────────────────────────

    def _add_videos(self):
        files = filedialog.askopenfilenames(
            filetypes=[("Video Files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm")]
        )
        for f in files:
            if f not in self.video_paths:
                self.video_paths.append(f)
        self._refresh_file_list()

    def _remove_selected(self):
        if self.video_paths:
            self.video_paths.pop()  # Remove last item
            self._refresh_file_list()

    def _clear_queue(self):
        self.video_paths.clear()
        self._refresh_file_list()

    def _refresh_file_list(self):
        """Rebuild the queue panel file rows."""
        for w in self.queue_scroll.winfo_children():
            w.destroy()

        if not self.video_paths:
            ctk.CTkLabel(
                self.queue_scroll, text="No videos added.\nUse '+ Add Videos' to begin.",
                font=(FONT_FAMILY, 11), text_color=COLORS["text_muted"],
            ).grid(row=0, column=0, columnspan=2, pady=30, sticky="ew")
            return

        for i, path in enumerate(self.video_paths):
            name = os.path.basename(path)
            # Truncate long names
            display_name = name if len(name) <= 22 else name[:19] + "..."

            row_bg = COLORS["bg_dark"] if i % 2 == 0 else COLORS["panel"]

            name_label = ctk.CTkLabel(
                self.queue_scroll, text=f"  📄 {display_name}",
                font=(FONT_FAMILY, 11), text_color=COLORS["text_secondary"],
                anchor="w",
            )
            name_label.grid(row=i, column=0, padx=(4, 0), pady=2, sticky="ew")

            # Status badge
            status_text = "Awaiting" if i == 0 else "Queue"
            status_color = COLORS["status_awaiting"] if i == 0 else COLORS["status_queue"]

            ctk.CTkLabel(
                self.queue_scroll, text=status_text,
                font=(FONT_FAMILY, 10, "bold"), text_color=status_color,
                anchor="e",
            ).grid(row=i, column=1, padx=(0, 8), pady=2, sticky="e")

    def _browse_output(self):
        dir_path = filedialog.askdirectory()
        if dir_path:
            self.output_dir = dir_path
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, dir_path)

    # ─── Processing ───────────────────────────────────────────────────────

    def _start_processing(self):
        if not self.video_paths:
            messagebox.showerror("Error", "Please add at least one video file.")
            return

        self.is_processing = True
        self.cancel_event.clear()
        self.start_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.progress_bar.set(0)
        self._log_clear()
        self._translation_log_clear()

        threading.Thread(target=self._processing_loop, daemon=True).start()

    def _cancel_processing(self):
        self.cancel_event.set()
        self._log("⚠  Cancellation requested... finishing current frame.")
        self.cancel_btn.configure(state="disabled")

    def _on_continue_clicked(self):
        """User approved the translations — signal pipeline to continue."""
        self.continue_event.set()
        self.after(0, lambda: self.continue_btn.pack_forget())
        self._translation_log("User approved translations. Continuing pipeline...")
        self._update_status("Resuming inpainting...")

    def _reconstruct_srt_from_editor(self):
        """Build an SRT string from the current Translation Editor data."""
        if not self.translation_data:
            return ""

        # Read latest values from entry widgets (user may have edited)
        for i, widget in enumerate(self.translation_entry_widgets):
            if i < len(self.translation_data):
                self.translation_data[i]["translated"] = widget.get()

        # Reconstruct SRT using original timestamps from source SRT
        source_blocks = re.split(r'\n\n+', self.current_srt_source.strip()) if self.current_srt_source else []

        srt_lines = []
        for i, entry in enumerate(self.translation_data):
            srt_lines.append(str(i + 1))
            # Use original timestamp line if available
            if i < len(source_blocks):
                block_lines = source_blocks[i].strip().split('\n')
                if len(block_lines) >= 2:
                    srt_lines.append(block_lines[1])  # Original timestamp
                else:
                    srt_lines.append(f"{entry['timestamp_start']}:00,000 --> {entry['timestamp_end']}:00,000")
            else:
                srt_lines.append(f"{entry['timestamp_start']}:00,000 --> {entry['timestamp_end']}:00,000")
            srt_lines.append(entry["translated"])
            srt_lines.append("")  # Blank line separator

        return "\n".join(srt_lines)

    def _show_continue_button(self):
        """Show the Continue Inpainting button in the editor toolbar."""
        self.continue_btn.pack(side="left", padx=(12, 0))

    def _hide_continue_button(self):
        """Hide the Continue Inpainting button."""
        self.continue_btn.pack_forget()

    def _processing_loop(self):
        """Process all queued videos with split-phase pipeline.
        
        Flow for subtitle modes:
          Phase 1: OCR/Whisper detection + AI Translation
          → PAUSE: Fill Translation Editor, wait for user review
          Phase 2: Inpainting + Rendering with user-edited subtitles
        
        Watermark mode runs straight through (no translation step).
        """
        target_name = self.target_lang_var.get()
        target_code = TARGET_LANGUAGES.get(target_name, "vi")
        translator_name = self.translator_var.get()
        # Map UI display name -> internal model identifier
        if translator_name == "ChatGPT":
            translator_model = "chatgpt"
        elif translator_name.startswith("Ollama"):
            translator_model = "ollama:" + translator_name.split("(")[1].rstrip(")")
        else:
            translator_model = "google"

        pipeline_mode = self.pipeline_mode_var.get()
        whisper_model = self.whisper_model_var.get()
        is_audio_mode = (pipeline_mode == "Audio Only (Whisper)")
        is_replace_mode = (pipeline_mode == "Replace Subs (Full)")
        is_watermark_mode = (pipeline_mode == "Watermark Removal")

        # Get encoding settings
        resolution_key = self.resolution_var.get()
        quality_key = self.quality_var.get()
        resolution_preset = RESOLUTION_PRESETS.get(resolution_key)
        quality_preset = QUALITY_PRESETS.get(quality_key, QUALITY_PRESETS["High"])

        total_videos = len(self.video_paths)
        output_dir = self.output_entry.get().strip() or None

        try:
            self._log(f"🚀  System: GPU Accelerated (RTX 3050 Check)")
            self._log(f"📐  Resolution: {resolution_key} | Quality: {quality_key}")
            if is_watermark_mode:
                self._log("🚀  Engine: AI Watermark Removal (LaMa Inpainting + NVENC)")
            elif is_replace_mode:
                self._log("🚀  Engine: Replace Subs — Inpaint + Whisper (medium)")
            elif is_audio_mode:
                self._log(f"🚀  Engine: Audio Transcription (Whisper {whisper_model})")
            else:
                self._log("🚀  Engine: Advanced Selective Inpainting (v4)")

            self._update_status("Processing...")

            for idx, video_path in enumerate(self.video_paths):
                if self.cancel_event.is_set():
                    break

                video_name = os.path.basename(video_path)
                self.after(0, lambda i=idx: self._update_queue_status(i, "Processing"))

                self._log(f"\n{'─' * 50}")
                self._log(f"📹  [{idx + 1}/{total_videos}] {video_name}")

                def progress_cb(msg, _idx=idx):
                    self._update_status(f"[{_idx + 1}/{total_videos}] {msg}")
                    self._log(f"   {msg}")
                    self._update_progress_from_msg(msg, _idx, total_videos)

                # ─────────── Watermark Mode (no translation) ──────────
                if is_watermark_mode:
                    self._log(f"   🤖  Running automatic temporal variance detection")
                    result = run_watermark_pipeline(
                        video_path,
                        progress_callback=progress_cb,
                        output_dir=output_dir,
                        regions=None,
                    )

                # ─────────── V4: Hardcoded Subs (OCR) ─────────────────
                elif not is_audio_mode and not is_replace_mode:
                    result = self._run_v4_split(
                        video_path, target_code, translator_model,
                        progress_cb, output_dir, idx, total_videos
                    )

                # ─────────── Audio Only (Whisper) ─────────────────────
                elif is_audio_mode:
                    result = self._run_audio_split(
                        video_path, target_code, translator_model,
                        whisper_model, progress_cb, output_dir, idx, total_videos
                    )

                # ─────────── Replace Subs (Full) ──────────────────────
                elif is_replace_mode:
                    result = self._run_replace_subs_split(
                        video_path, target_code, translator_model,
                        progress_cb, output_dir, idx, total_videos
                    )

                if result:
                    self._log(f"   ✅  Saved → {os.path.basename(result)}")
                    self.after(0, lambda i=idx: self._update_queue_status(i, "Done"))
                elif self.cancel_event.is_set():
                    self._log(f"   ❌  Cancelled")
                    break

                self.after(0, lambda: self.eta_label.configure(text="Remaining: ---"))

            if self.cancel_event.is_set():
                self._update_status("Cancelled.")
                self._log(f"\n⚠  Processing cancelled by user.")
            else:
                self.after(0, lambda: self.progress_bar.set(1.0))
                self.after(0, lambda: self.progress_pct_label.configure(text="100%"))
                self._update_status("All done!")
                self._log(f"\n🎉  All {total_videos} video(s) processed successfully!")
                self.after(0, lambda: messagebox.showinfo("Success", "All videos processed!"))

        except Exception as e:
            err_msg = str(e)
            self._update_status(f"Error: {err_msg}")
            self._log(f"\n❌  Error: {err_msg}")
            self.after(0, lambda em=err_msg: messagebox.showerror("Error", em))

        finally:
            self.is_processing = False
            self.after(0, lambda: self.start_btn.configure(state="normal"))
            self.after(0, lambda: self.cancel_btn.configure(state="disabled"))
            self.after(0, self._hide_continue_button)

    # ─── Split-Phase Pipeline Methods ─────────────────────────────────────

    def _wait_for_user_review(self, srt_source, srt_translated):
        """Pause pipeline, fill Translation Editor, wait for user to click Continue.
        
        Returns the (possibly edited) translated SRT string.
        """
        self.continue_event.clear()

        # Fill the editor on the main thread
        self.after(0, lambda s=srt_source, t=srt_translated: self._load_translation_data(s, t))
        self.after(0, self._show_continue_button)

        self._update_status("⏸ Waiting for translation review...")
        self._log("   ⏸ Translation Editor loaded — review and edit, then click '▶ Continue Inpainting'")
        self._translation_log("Pipeline paused. Edit translations, then click Continue.")

        # Block this thread until user clicks Continue or cancels
        while not self.continue_event.is_set() and not self.cancel_event.is_set():
            self.continue_event.wait(timeout=0.5)

        if self.cancel_event.is_set():
            return None

        # Reconstruct SRT from editor (must read widgets on main thread)
        result_holder = [None]
        done_event = threading.Event()

        def _read_editor():
            result_holder[0] = self._reconstruct_srt_from_editor()
            done_event.set()

        self.after(0, _read_editor)
        done_event.wait(timeout=10)

        edited_srt = result_holder[0] or srt_translated
        self._translation_log(f"Editor data collected: {len(self.translation_data)} entries")
        return edited_srt

    def _run_v4_split(self, video_path, target_code, translator_model,
                      progress_cb, output_dir, idx, total_videos):
        """V4 pipeline split into: OCR+Translate → User Review → Inpaint+Render."""
        # Phase 1: OCR Detection
        self._log("   Phase 1: OCR Detection + Translation")
        pipe = SelectiveInpaintPipe()
        ocr_history, fps = pipe.extract_metadata(video_path, progress_cb)

        segments = get_stabilized_segments(ocr_history, fps)
        self._log(f"   OCR detected text on {len(ocr_history)} frames → {len(segments)} segments")

        if not segments:
            self._log("   ⚠ No subtitles detected — skipping")
            return None

        # Phase 1b: Translation
        original_srt = frames_to_srt(ocr_history, fps)
        translated_srt = pipe.clean_and_translate_srt(
            ocr_history, fps, target_code, translator_model, progress_cb
        )

        if not translated_srt or not translated_srt.strip():
            self._log("   ⚠ Translation returned empty")
            translated_srt = original_srt

        # ── PAUSE: User review in Translation Editor ──
        edited_srt = self._wait_for_user_review(original_srt, translated_srt)
        if edited_srt is None:  # Cancelled
            return None

        # Phase 2: Inpainting + Rendering with edited subtitles
        self._log("   Phase 2: AI Inpainting + Subtitle Rendering")
        result = pipe.inpaint_and_render(
            video_path, segments, edited_srt,
            progress_callback=progress_cb, output_dir=output_dir
        )
        return result

    def _run_audio_split(self, video_path, target_code, translator_model,
                         whisper_model, progress_cb, output_dir, idx, total_videos):
        """Audio pipeline split into: Whisper+Translate → User Review → Render."""
        # Phase 1a: Extract audio
        self._log("   Phase 1: Audio Extraction + Transcription")
        audio_path = extract_audio(video_path, progress_cb)
        if not audio_path:
            self._log("   ❌ Audio extraction failed")
            return None

        # Phase 1b: Transcribe
        srt_content, detected_lang = transcribe_audio(audio_path, whisper_model, progress_cb)

        # Clean up temp audio
        if os.path.exists(audio_path):
            os.remove(audio_path)

        if not srt_content or not srt_content.strip():
            self._log("   ⚠ No speech detected")
            return None

        # Phase 1c: Translate
        self._log(f"   Translating to {target_code}...")
        translator = AITranslator(model=translator_model)
        translated_srt = translator.translate_srt_content(
            srt_content, target_code, progress_callback=progress_cb
        )
        translator.unload()

        if not translated_srt or not translated_srt.strip():
            translated_srt = srt_content

        # ── PAUSE: User review in Translation Editor ──
        edited_srt = self._wait_for_user_review(srt_content, translated_srt)
        if edited_srt is None:
            return None

        # Phase 2: Render subtitles onto video
        self._log("   Phase 2: Rendering subtitles onto video")
        result = render_subtitles(video_path, edited_srt, progress_cb, output_dir=output_dir)
        return result

    def _run_replace_subs_split(self, video_path, target_code, translator_model,
                                progress_cb, output_dir, idx, total_videos):
        """Replace Subs pipeline split into:
        OCR+Inpaint old → Whisper+Translate → User Review → Render new."""
        # Phase 1a: OCR detect old subtitles
        self._log("   Phase 1: Detecting and removing old subtitles")
        pipe = SelectiveInpaintPipe()
        ocr_history, fps = pipe.extract_metadata(video_path, progress_cb)

        segments = get_stabilized_segments(ocr_history, fps)
        self._log(f"   OCR detected {len(segments)} subtitle segments")

        if not segments:
            self._log("   ⚠ No old subs found — falling back to audio-only flow")
            return self._run_audio_split(
                video_path, target_code, translator_model,
                "medium", progress_cb, output_dir, idx, total_videos
            )

        # Phase 1b: Inpaint (remove old subs) — no new subs yet
        clean_video = pipe.inpaint_and_render(
            video_path, segments, translated_srt="",
            progress_callback=progress_cb,
        )

        # Free GPU memory
        del pipe
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

        if not clean_video or not os.path.exists(clean_video):
            self._log("   ❌ Inpainting failed")
            return None

        # Phase 1c: Extract audio + Whisper transcribe
        self._log("   Extracting audio and transcribing...")
        audio_path = extract_audio(video_path, progress_cb)
        if not audio_path:
            self._log("   ❌ Audio extraction failed")
            return clean_video

        srt_content, detected_lang = transcribe_audio(audio_path, "medium", progress_cb)
        if os.path.exists(audio_path):
            os.remove(audio_path)

        if not srt_content or not srt_content.strip():
            self._log("   ⚠ No speech detected — returning clean video")
            return clean_video

        # Phase 1d: Translate
        self._log(f"   Translating to {target_code}...")
        translator = AITranslator(model=translator_model)
        translated_srt = translator.translate_srt_content(
            srt_content, target_code, progress_callback=progress_cb
        )
        translator.unload()

        if not translated_srt or not translated_srt.strip():
            translated_srt = srt_content

        # ── PAUSE: User review in Translation Editor ──
        edited_srt = self._wait_for_user_review(srt_content, translated_srt)
        if edited_srt is None:
            return clean_video

        # Phase 2: Render new subs onto clean video
        self._log("   Phase 2: Rendering new subtitles onto clean video")
        result = render_subtitles(clean_video, edited_srt, progress_cb, output_dir=output_dir)

        # Clean up intermediate video
        if result and os.path.exists(result) and result != clean_video:
            try:
                os.remove(clean_video)
            except OSError:
                pass

        return result

    def _update_queue_status(self, index, status):
        """Update a specific file's status badge in the queue."""
        children = self.queue_scroll.winfo_children()
        # Each file has 2 widgets (name + status), so status is at index*2 + 1
        widget_idx = index * 2 + 1
        if widget_idx < len(children):
            status_colors = {
                "Processing": COLORS["accent"],
                "Done": COLORS["status_done"],
                "Awaiting": COLORS["status_awaiting"],
                "Queue": COLORS["status_queue"],
            }
            children[widget_idx].configure(
                text=status,
                text_color=status_colors.get(status, COLORS["text_muted"])
            )

    def _update_progress_from_msg(self, msg, video_idx, total_videos):
        """Try to extract a percentage from the progress message and update the bar."""
        if not msg or not isinstance(msg, str): return
        try:
            video_share = 1.0 / total_videos
            base_overall = video_idx * video_share
            
            if "%" in msg:
                parts = msg.split("%")[0].split()
                if parts:
                    pct_str = parts[-1].rstrip("%")
                    step_pct = int(pct_str) / 100.0
                    
                    if "OCR Detection" in msg:
                        overall = base_overall + (step_pct * 0.3 * video_share)
                    elif "Translating" in msg:
                        overall = base_overall + (0.3 * video_share) + (step_pct * 0.15 * video_share)
                    elif "Inpainting" in msg and "Rendering" in msg:
                        overall = base_overall + (0.20 * video_share) + (step_pct * 0.30 * video_share)
                    elif "Inpainting" in msg or "Rendering" in msg:
                        if "Rendering subtitles" in msg:
                            overall = base_overall + (0.85 * video_share) + (step_pct * 0.15 * video_share)
                        else:
                            overall = base_overall + (0.45 * video_share) + (step_pct * 0.55 * video_share)
                    elif "Transcribing" in msg:
                        overall = base_overall + (step_pct * 0.5 * video_share)
                    else:
                        overall = base_overall + (step_pct * video_share)
                        
                    self.after(0, lambda p=overall: self.progress_bar.set(p))
                    self.after(0, lambda p=int(overall * 100): self.progress_pct_label.configure(text=f"{p}%"))
            
            if "ETA:" in msg:
                eta_val = msg.split("ETA:")[-1].strip()
                self.after(0, lambda e=eta_val: self.eta_label.configure(text=f"Remaining: {e}"))

            if "Merging" in msg or "Writing" in msg:
                overall = ((video_idx + 0.95) * video_share)
                self.after(0, lambda p=overall: self.progress_bar.set(p))
                self.after(0, lambda p=int(overall * 100): self.progress_pct_label.configure(text=f"{p}%"))
        except Exception:
            pass

    # ─── Helpers ──────────────────────────────────────────────────────────

    def _update_status(self, msg):
        if msg is None: msg = ""
        self.after(0, lambda: self.status_label.configure(text=str(msg)))

    def _log(self, msg):
        def _append():
            self.log_console.configure(state="normal")
            self.log_console.insert("end", msg + "\n")
            self.log_console.see("end")
            self.log_console.configure(state="disabled")
        self.after(0, _append)

    def _log_clear(self):
        def _clear():
            self.log_console.configure(state="normal")
            self.log_console.delete("1.0", "end")
            self.log_console.configure(state="disabled")
        self.after(0, _clear)

    def _translation_log(self, msg):
        """Log to the translation (right) log panel."""
        import datetime
        timestamp = datetime.datetime.now().strftime("[%H:%M:%S]")
        def _append():
            self.translation_log_console.configure(state="normal")
            self.translation_log_console.insert("end", f"{timestamp} {msg}\n")
            self.translation_log_console.see("end")
            self.translation_log_console.configure(state="disabled")
        self.after(0, _append)

    def _translation_log_clear(self):
        def _clear():
            self.translation_log_console.configure(state="normal")
            self.translation_log_console.delete("1.0", "end")
            self.translation_log_console.configure(state="disabled")
        self.after(0, _clear)


if __name__ == "__main__":
    app = App()
    app.mainloop()
