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
from pipeline_v4 import run_v4

# ─── Color Palette & Theme ───────────────────────────────────────────────────

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
    "progress_bg": "#1e1e2e",
    "progress_fill": "#7c3aed",
}

FONT_FAMILY = "Segoe UI"

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")


# ─── Supported Languages ─────────────────────────────────────────────────────

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


class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("VietSub — Video Subtitle Translator")
        self.geometry("960x700")
        self.minsize(860, 620)
        self.configure(fg_color=COLORS["bg_dark"])

        self.video_paths = []
        self.output_dir = None
        self.cancel_event = threading.Event()
        self.is_processing = False

        # ── Layout: sidebar + main ──────────────────────────────────────
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._build_sidebar()
        self._build_main_area()

    # ─── Sidebar ──────────────────────────────────────────────────────────

    def _build_sidebar(self):
        sidebar = ctk.CTkFrame(self, width=220, corner_radius=0, fg_color=COLORS["sidebar"])
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(10, weight=1)  # Push bottom items down

        # Logo / title
        ctk.CTkLabel(
            sidebar, text="🎬 VietSub", font=(FONT_FAMILY, 22, "bold"),
            text_color=COLORS["accent_light"],
        ).grid(row=0, column=0, padx=20, pady=(25, 2), sticky="w")

        ctk.CTkLabel(
            sidebar, text="Subtitle Translator & Upscaler",
            font=(FONT_FAMILY, 11), text_color=COLORS["text_muted"],
        ).grid(row=1, column=0, padx=20, pady=(0, 20), sticky="w")

        sep = ctk.CTkFrame(sidebar, height=1, fg_color=COLORS["border"])
        sep.grid(row=2, column=0, sticky="ew", padx=15, pady=5)

        ctk.CTkLabel(
            sidebar, text="TRANSLATOR MODEL", font=(FONT_FAMILY, 10, "bold"),
            text_color=COLORS["text_muted"],
        ).grid(row=2, column=0, padx=20, pady=(15, 2), sticky="w")
        
        self.translator_var = ctk.StringVar(value="Google Translate")
        ctk.CTkOptionMenu(
            sidebar, variable=self.translator_var,
            values=["Google Translate", "ChatGPT"],
            fg_color=COLORS["card"], button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
            dropdown_fg_color=COLORS["card"],
            width=180,
        ).grid(row=3, column=0, padx=20, pady=(0, 8), sticky="w")

        # ── Target language ──
        ctk.CTkLabel(
            sidebar, text="TARGET LANGUAGE", font=(FONT_FAMILY, 10, "bold"),
            text_color=COLORS["text_muted"],
        ).grid(row=4, column=0, padx=20, pady=(15, 4), sticky="w")

        self.target_lang_var = ctk.StringVar(value="Vietnamese")
        ctk.CTkOptionMenu(
            sidebar, variable=self.target_lang_var,
            values=list(TARGET_LANGUAGES.keys()),
            fg_color=COLORS["card"], button_color=COLORS["accent"],
            button_hover_color=COLORS["accent_hover"],
            dropdown_fg_color=COLORS["card"],
            width=180,
        ).grid(row=5, column=0, padx=20, pady=(0, 8), sticky="w")

        sep2 = ctk.CTkFrame(sidebar, height=1, fg_color=COLORS["border"])
        sep2.grid(row=6, column=0, sticky="ew", padx=15, pady=10)

    # ─── Main Area ────────────────────────────────────────────────────────

    def _build_main_area(self):
        main = ctk.CTkFrame(self, fg_color="transparent")
        main.grid(row=0, column=1, sticky="nsew", padx=15, pady=15)
        main.grid_columnconfigure(0, weight=1)
        main.grid_rowconfigure(2, weight=1)  # Log console expands
        main.grid_rowconfigure(1, weight=2)  # File list expands more

        # ── File Queue Card ──────────────────────────────────────────────
        file_card = ctk.CTkFrame(main, fg_color=COLORS["card"], corner_radius=12)
        file_card.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        file_card.grid_columnconfigure(0, weight=1)

        header_row = ctk.CTkFrame(file_card, fg_color="transparent")
        header_row.grid(row=0, column=0, sticky="ew", padx=16, pady=(14, 6))
        header_row.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(
            header_row, text="📂  Video Queue", font=(FONT_FAMILY, 15, "bold"),
            text_color=COLORS["text_primary"],
        ).grid(row=0, column=0, sticky="w")

        btn_frame = ctk.CTkFrame(header_row, fg_color="transparent")
        btn_frame.grid(row=0, column=1, sticky="e")

        ctk.CTkButton(
            btn_frame, text="+ Add Videos", width=110, height=30,
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            font=(FONT_FAMILY, 12, "bold"), command=self._add_videos,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            btn_frame, text="Remove", width=80, height=30,
            fg_color=COLORS["danger"], hover_color=COLORS["danger_hover"],
            font=(FONT_FAMILY, 12), command=self._remove_selected,
        ).pack(side="left", padx=(0, 6))

        ctk.CTkButton(
            btn_frame, text="Clear All", width=80, height=30,
            fg_color=COLORS["card_hover"], hover_color=COLORS["border"],
            font=(FONT_FAMILY, 12), command=self._clear_queue,
        ).pack(side="left")

        # File listbox (using CTkTextbox as a selectable list)
        self.file_listbox = ctk.CTkTextbox(
            file_card, height=120, font=(FONT_FAMILY, 12),
            fg_color=COLORS["bg_dark"], text_color=COLORS["text_secondary"],
            corner_radius=8, border_width=1, border_color=COLORS["border"],
        )
        self.file_listbox.grid(row=1, column=0, sticky="ew", padx=16, pady=(0, 14))
        self.file_listbox.configure(state="disabled")

        # ── Output Directory ──
        out_row = ctk.CTkFrame(file_card, fg_color="transparent")
        out_row.grid(row=2, column=0, sticky="ew", padx=16, pady=(0, 14))
        out_row.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(
            out_row, text="Output:", font=(FONT_FAMILY, 12, "bold"),
            text_color=COLORS["text_muted"],
        ).grid(row=0, column=0, padx=(0, 8), sticky="w")

        self.output_entry = ctk.CTkEntry(
            out_row, placeholder_text="Same as input video...",
            fg_color=COLORS["bg_dark"], border_color=COLORS["border"],
            text_color=COLORS["text_primary"],
        )
        self.output_entry.grid(row=0, column=1, sticky="ew", padx=(0, 8))

        ctk.CTkButton(
            out_row, text="Browse", width=80, height=30,
            fg_color=COLORS["card_hover"], hover_color=COLORS["border"],
            font=(FONT_FAMILY, 12), command=self._browse_output,
        ).grid(row=0, column=2)

        # ── Progress Section ─────────────────────────────────────────────
        progress_card = ctk.CTkFrame(main, fg_color=COLORS["card"], corner_radius=12)
        progress_card.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        progress_card.grid_columnconfigure(0, weight=1)

        self.status_label = ctk.CTkLabel(
            progress_card, text="Ready", font=(FONT_FAMILY, 13),
            text_color=COLORS["text_secondary"], anchor="w",
        )
        self.status_label.grid(row=0, column=0, padx=16, pady=(14, 6), sticky="ew")

        self.progress_bar = ctk.CTkProgressBar(
            progress_card, height=10, corner_radius=5,
            fg_color=COLORS["progress_bg"], progress_color=COLORS["progress_fill"],
        )
        self.progress_bar.grid(row=1, column=0, padx=16, pady=(0, 6), sticky="ew")
        self.progress_bar.set(0)

        self.progress_pct_label = ctk.CTkLabel(
            progress_card, text="0%", font=(FONT_FAMILY, 11, "bold"),
            text_color=COLORS["accent_light"],
        )
        self.progress_pct_label.grid(row=1, column=0, padx=16, pady=(0, 6), sticky="e")

        self.eta_label = ctk.CTkLabel(
            progress_card, text="", font=(FONT_FAMILY, 11),
            text_color=COLORS["text_muted"],
        )
        self.eta_label.grid(row=0, column=0, padx=16, pady=(14, 6), sticky="e")

        # Action buttons
        btn_row = ctk.CTkFrame(progress_card, fg_color="transparent")
        btn_row.grid(row=2, column=0, padx=16, pady=(4, 14), sticky="ew")
        btn_row.grid_columnconfigure(0, weight=1)

        self.start_btn = ctk.CTkButton(
            btn_row, text="▶  Start Processing", height=42,
            fg_color=COLORS["accent"], hover_color=COLORS["accent_hover"],
            font=(FONT_FAMILY, 14, "bold"), command=self._start_processing,
        )
        self.start_btn.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        self.cancel_btn = ctk.CTkButton(
            btn_row, text="✕  Cancel", height=42, width=120,
            fg_color=COLORS["danger"], hover_color=COLORS["danger_hover"],
            font=(FONT_FAMILY, 14, "bold"), command=self._cancel_processing,
            state="disabled",
        )
        self.cancel_btn.grid(row=0, column=1, sticky="e")

        # ── Log Console ──────────────────────────────────────────────────
        log_card = ctk.CTkFrame(main, fg_color=COLORS["card"], corner_radius=12)
        log_card.grid(row=2, column=0, sticky="nsew")
        log_card.grid_columnconfigure(0, weight=1)
        log_card.grid_rowconfigure(1, weight=1)

        ctk.CTkLabel(
            log_card, text="📋  Processing Log", font=(FONT_FAMILY, 13, "bold"),
            text_color=COLORS["text_primary"],
        ).grid(row=0, column=0, padx=16, pady=(12, 4), sticky="w")

        self.log_console = ctk.CTkTextbox(
            log_card, font=("Consolas", 11),
            fg_color=COLORS["bg_dark"], text_color=COLORS["text_muted"],
            corner_radius=8, border_width=1, border_color=COLORS["border"],
        )
        self.log_console.grid(row=1, column=0, sticky="nsew", padx=16, pady=(0, 14))
        self.log_console.configure(state="disabled")

    # (Unused slider callbacks removed)


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
        self.file_listbox.configure(state="normal")
        self.file_listbox.delete("1.0", "end")
        for i, path in enumerate(self.video_paths, 1):
            name = os.path.basename(path)
            self.file_listbox.insert("end", f"  {i}. {name}\n")
        if not self.video_paths:
            self.file_listbox.insert("end", "  No videos added. Click '+ Add Videos' to begin.")
        self.file_listbox.configure(state="disabled")

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

        threading.Thread(target=self._processing_loop, daemon=True).start()

    def _cancel_processing(self):
        self.cancel_event.set()
        self._log("⚠  Cancellation requested... finishing current frame.")
        self.cancel_btn.configure(state="disabled")

    def _processing_loop(self):
        """Process all queued videos sequentially."""
        target_name = self.target_lang_var.get()
        target_code = TARGET_LANGUAGES.get(target_name, "vi")
        translator_name = self.translator_var.get()
        translator_model = "chatgpt" if translator_name == "ChatGPT" else "google"

        total_videos = len(self.video_paths)

        try:
            self._log(f"🚀  System: GPU Accelerated (RTX 3050 Check)")
            self._log("🚀  Engine: Advanced Selective Inpainting (v4)")

            for idx, video_path in enumerate(self.video_paths):
                if self.cancel_event.is_set():
                    break

                video_name = os.path.basename(video_path)
                
                self._log(f"\n{'─' * 50}")
                self._log(f"📹  [{idx + 1}/{total_videos}] {video_name}")

                def progress_cb(msg, _idx=idx):
                    self._update_status(f"[{_idx + 1}/{total_videos}] {msg}")
                    self._log(f"   {msg}")
                    self._update_progress_from_msg(msg, _idx, total_videos)

                # Logic strictly for v4 pipeline
                result = run_v4(
                    video_path, 
                    target_code,
                    translator_model=translator_model,
                    progress_callback=progress_cb
                )

                if result:
                    self._log(f"   ✅  Saved → {os.path.basename(result)}")
                elif self.cancel_event.is_set():
                    self._log(f"   ❌  Cancelled")
                    break
                
                # Clear ETA after each video
                self.after(0, lambda: self.eta_label.configure(text=""))

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
            self._update_status(f"Error: {e}")
            self._log(f"\n❌  Error: {e}")
            self.after(0, lambda: messagebox.showerror("Error", str(e)))

        finally:
            self.is_processing = False
            self.after(0, lambda: self.start_btn.configure(state="normal"))
            self.after(0, lambda: self.cancel_btn.configure(state="disabled"))

    def _update_progress_from_msg(self, msg, video_idx, total_videos):
        """Try to extract a percentage from the progress message and update the bar."""
        if not msg or not isinstance(msg, str): return
        try:
            if "%" in msg:
                # Extract the percentage number before the % sign
                parts = msg.split("%")[0].split()
                if parts:
                    pct_str = parts[-1].rstrip("%")
                    frame_pct = int(pct_str) / 100.0
                    # Scale to overall progress across all videos
                    video_share = 1.0 / total_videos
                    overall = (video_idx * video_share) + (frame_pct * video_share)
                    self.after(0, lambda p=overall: self.progress_bar.set(p))
                    self.after(0, lambda p=int(overall * 100): self.progress_pct_label.configure(text=f"{p}%"))
            
            # Extract ETA if present
            if "ETA:" in msg:
                eta_val = msg.split("ETA:")[-1].strip()
                self.after(0, lambda e=eta_val: self.eta_label.configure(text=f"ETA: {e}"))

            if "Merging" in msg or "Writing" in msg:
                video_share = 1.0 / total_videos
                overall = ((video_idx + 0.9) * video_share)
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


if __name__ == "__main__":
    app = App()
    app.mainloop()
