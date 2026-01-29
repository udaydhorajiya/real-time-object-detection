# ==============================================================================
# 1. IMPORTS
# ==============================================================================
import os
import queue
import threading
import time
import tkinter as tk
from tkinter import Button, Label, Canvas

import cv2
import pygame
import torch
from PIL import Image, ImageTk
from ultralytics import YOLO


# ==============================================================================
# 2. APPLICATION CLASS
# ==============================================================================
class ObjectDetectorApp:
    """
    A real-time, full-screen object detection application with a graphical user interface
    built using Tkinter, OpenCV, and a YOLO model.
    """

    def __init__(self, root):
        """Initializes the main application window and all necessary variables."""
        self.root = root
        self.root.title("Real-Time Object Detector")

        # --- Window and Screen Setup ---
        self.screen_width = self.root.winfo_screenwidth()
        self.screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{self.screen_width}x{self.screen_height}+0+0")

        # --- Core State Variables ---
        self.is_detection_running = False
        self.is_bg_video_running = False
        self.model = None
        self.cap = None

        # --- UI Element Variables ---
        self.canvas = None
        self.video_label = None
        self.stop_button = None

        # --- Threading and Queues ---
        self.detection_thread = None
        self.bg_video_thread = None
        self.frame_queue = queue.Queue(maxsize=1)
        self.bg_frame_queue = queue.Queue(maxsize=1)

        # --- Animation Variables ---
        self.bg_animation_job = None
        self.pulse_animation_job = None
        self.title_animation_job = None
        self.pulse_direction = 1
        self.pulse_color_value = 120
        self.full_title_text = "Real Time Object Detection"
        self.title_char_index = 0

        # --- Alert System Variables ---
        self.alert_active = False
        self.alert_sound = None

        # --- UI Styling ---
        self.primary_color = "#0078D7"
        self.hover_color = "#0098FF"

        # --- Initializations ---
        self._initialize_sound()
        self.create_start_screen()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    # --------------------------------------------------------------------------
    # 3. INITIALIZATION & UI SETUP METHODS
    # --------------------------------------------------------------------------

    def _initialize_sound(self):
        """Initializes the pygame mixer and loads the alert sound if available."""
        try:
            pygame.mixer.init()
            sound_path = 'alert.mp3'
            if os.path.exists(sound_path):
                self.alert_sound = pygame.mixer.Sound(sound_path)
                print("Custom alert sound loaded successfully.")
            else:
                print(f"Warning: Custom alert sound '{sound_path}' not found. Using system beep instead.")
        except Exception as e:
            print(f"Error initializing sound system or loading sound: {e}")
            print("Will fall back to system beep.")

    def create_start_screen(self):
        """Creates the main start screen with animations and interactive elements."""
        self.canvas = Canvas(self.root, width=self.screen_width, height=self.screen_height, highlightthickness=0)
        self.canvas.pack()

        self._setup_background()
        self._setup_title()
        self._setup_start_button()

        self.animate_button_pulse()
        self.animate_title_typewriter()

    def _setup_background(self):
        """Sets up the animated video background or a fallback static image."""
        try:
            video_path = "C:/Users/henil/Downloads/videoplayback.webm"
            self.bg_video_cap = cv2.VideoCapture(video_path)
            if not self.bg_video_cap.isOpened():
                raise FileNotFoundError

            self.is_bg_video_running = True
            self.bg_video_thread = threading.Thread(target=self._background_video_loop, daemon=True)
            self.bg_video_thread.start()
            self.canvas.create_image(0, 0, anchor='nw', tags="background")
            self.animate_video_display()
        except (FileNotFoundError, Exception):
            print("Warning: Animated background not found. Falling back to static image.")
            if self.bg_video_cap: self.bg_video_cap.release()
            try:
                bg_image = Image.open("img2.jpg")
                bg_image = bg_image.resize((self.screen_width, self.screen_height), Image.Resampling.LANCZOS)
                self.bg_photo = ImageTk.PhotoImage(bg_image)
                self.canvas.create_image(0, 0, image=self.bg_photo, anchor='nw')
            except FileNotFoundError:
                print("Error: Static background image not found. Using plain background.")
                self.canvas.configure(bg='grey')

    def _setup_title(self):
        """Sets up the animated title text on the canvas."""
        title_font = ("Movavi Grotesque Black", 40, "bold")
        x_center, y_center = self.screen_width // 2, self.screen_height * 0.2
        self.canvas.create_text(x_center, y_center, text="", font=title_font, fill="white", anchor="center",
                                tags="main_title")

    def _setup_start_button(self):
        """Creates the interactive 'Start Detection' button on the canvas."""
        button_font = ("Helvetica", 16, "bold")
        btn_x, btn_y = self.screen_width // 2, int(self.screen_height * 0.6)
        btn_width, btn_height = 220, 60
        radius = 20

        self.create_rounded_rectangle(btn_x - btn_width / 2, btn_y - btn_height / 2, btn_x + btn_width / 2,
                                      btn_y + btn_height / 2,
                                      radius=radius, fill=self.primary_color, tags="start_button")
        self.canvas.create_text(btn_x, btn_y, text="Start Detection", font=button_font, fill="white",
                                tags="start_button")

        self.canvas.tag_bind("start_button", "<Button-1>", lambda e: self.start_detection())
        self.canvas.tag_bind("start_button", "<Enter>", self.on_button_enter)
        self.canvas.tag_bind("start_button", "<Leave>", self.on_button_leave)

    def create_rounded_rectangle(self, x1, y1, x2, y2, radius=25, **kwargs):
        """Helper function to draw a rounded rectangle on the canvas."""
        points = [x1 + radius, y1, x1 + radius, y1, x2 - radius, y1, x2 - radius, y1, x2, y1, x2, y1 + radius,
                  x2, y1 + radius, x2, y2 - radius, x2, y2 - radius, x2, y2, x2 - radius, y2, x2 - radius, y2,
                  x1 + radius, y2, x1 + radius, y2, x1, y2, x1, y2 - radius, x1, y2 - radius, x1, y1 + radius,
                  x1, y1 + radius, x1, y1]
        return self.canvas.create_polygon(points, **kwargs, smooth=True)

    # --------------------------------------------------------------------------
    # 4. UI ANIMATION METHODS
    # --------------------------------------------------------------------------

    def animate_video_display(self):
        """(UI Thread) Displays frames from the background video queue."""
        if self.is_bg_video_running and self.canvas.winfo_exists():
            try:
                frame_rgb = self.bg_frame_queue.get_nowait()
                img_pil = Image.fromarray(frame_rgb)
                self.current_bg_frame_tk = ImageTk.PhotoImage(image=img_pil)
                bg_item = self.canvas.find_withtag("background")
                if bg_item:
                    self.canvas.itemconfig(bg_item[0], image=self.current_bg_frame_tk)
            except queue.Empty:
                pass
            self.bg_animation_job = self.root.after(30, self.animate_video_display)

    def animate_button_pulse(self):
        """Animates the start button with a pulsing color effect."""
        if self.canvas.winfo_exists():
            if self.pulse_color_value >= 200:
                self.pulse_direction = -1
            elif self.pulse_color_value <= 120:
                self.pulse_direction = 1
            self.pulse_color_value += self.pulse_direction * 2
            pulse_color = f'#00{self.pulse_color_value:02x}{255:02x}'
            shape_item = self.canvas.find_withtag("start_button")
            if shape_item:
                self.canvas.itemconfig(shape_item[0], fill=pulse_color)
            self.pulse_animation_job = self.root.after(20, self.animate_button_pulse)

    def animate_title_typewriter(self):
        """Animates the main title with a typewriter effect."""
        if self.canvas.winfo_exists():
            if self.title_char_index > len(self.full_title_text):
                self.title_char_index = 0
                delay = 2000  # Pause before restart
            else:
                display_text = self.full_title_text[:self.title_char_index]
                title_item = self.canvas.find_withtag("main_title")
                if title_item:
                    self.canvas.itemconfig(title_item[0], text=display_text)
                self.title_char_index += 1
                delay = 150  # Delay between characters
            self.title_animation_job = self.root.after(delay, self.animate_title_typewriter)

    # --------------------------------------------------------------------------
    # 5. UI EVENT HANDLERS
    # --------------------------------------------------------------------------

    def on_button_enter(self, e):
        """Handles the mouse entering the start button."""
        if self.pulse_animation_job:
            self.root.after_cancel(self.pulse_animation_job)
            self.pulse_animation_job = None
        shape_item = self.canvas.find_withtag("start_button")[0]
        self.canvas.itemconfig(shape_item, fill=self.hover_color)

    def on_button_leave(self, e):
        """Handles the mouse leaving the start button."""
        shape_item = self.canvas.find_withtag("start_button")[0]
        self.canvas.itemconfig(shape_item, fill=self.primary_color)
        if not self.pulse_animation_job:
            self.animate_button_pulse()

    # --------------------------------------------------------------------------
    # 6. CORE APPLICATION LOGIC
    # --------------------------------------------------------------------------

    def start_detection(self):
        """Stops all start-screen animations and transitions to the detection view."""
        self._stop_all_animations()
        if self.canvas: self.canvas.destroy(); self.canvas = None

        self.video_label = Label(self.root)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        self.stop_button = Button(self.root, text="Stop Detection", command=self.stop_detection,
                                  font=("Helvetica", 14, "bold"), bg=self.primary_color, fg="white",
                                  activebackground=self.hover_color, activeforeground="white", relief=tk.FLAT)
        self.stop_button.place(relx=0.5, rely=0.95, anchor="center")

        self.is_detection_running = True
        self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
        self.detection_thread.start()
        self.root.after(10, self.update_frame)

    def stop_detection(self):
        """Stops the detection process and returns to the start screen."""
        print("Stopping detection...")
        self.is_detection_running = False
        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1)

        if self.alert_sound and self.alert_active:
            self.alert_sound.stop()
            self.alert_active = False

        if self.video_label: self.video_label.destroy()
        if self.stop_button: self.stop_button.destroy()

        self.create_start_screen()

    def load_model(self):
        """Loads the YOLO model, falling back to a default if necessary."""
        model_path = 'yolov8l.pt'
        if not os.path.exists(model_path):
            print(f"'{model_path}' not found. Falling back to 'yolov8n.pt'.")
            model_path = 'yolov8n.pt'

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Loading model '{model_path}' onto device: {device}")
        try:
            self.model = YOLO(model_path)
            self.model.to(device)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            self.on_closing()

    # --------------------------------------------------------------------------
    # 7. THREADING LOOPS
    # --------------------------------------------------------------------------

    def _background_video_loop(self):
        """(Worker Thread) Reads and processes background video frames."""
        video_fps = self.bg_video_cap.get(cv2.CAP_PROP_FPS)
        delay = 1 / video_fps if video_fps > 0 else 0.03

        while self.is_bg_video_running:
            success, frame = self.bg_video_cap.read()
            if not success:
                self.bg_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            frame_resized = cv2.resize(frame, (self.screen_width, self.screen_height))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

            try:
                if not self.bg_frame_queue.empty(): self.bg_frame_queue.get_nowait()
                self.bg_frame_queue.put(frame_rgb, block=False)
            except queue.Full:
                continue
            time.sleep(delay)

    def _detection_loop(self):
        """(Worker Thread) Handles webcam capture and model inference."""
        self.load_model()
        if self.model is None: return

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open webcam.")
            return

        print("Webcam started. Detection running...")
        start_time = time.time()
        frame_count = 0
        fps = 0

        while self.is_detection_running:
            success, frame = self.cap.read()
            if not success:
                time.sleep(0.01)
                continue

            frame_resized = cv2.resize(frame, (640, 480))
            results = self.model(frame_resized, stream=True, verbose=False)
            harmful_object_detected = False

            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = round(float(box.conf[0]), 2)
                    class_name = self.model.names[int(box.cls[0])]

                    box_color = (0, 0, 255) if class_name in ["knife", "scissors"] else (0, 255, 0)
                    if class_name in ["knife", "scissors"]: harmful_object_detected = True

                    cv2.rectangle(frame_resized, (x1, y1), (x2, y2), box_color, 2)
                    label = f'{class_name}: {confidence}'
                    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    cv2.rectangle(frame_resized, (x1, y1 - h - 10), (x1 + w, y1), box_color, cv2.FILLED)
                    cv2.putText(frame_resized, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

            if harmful_object_detected and not self.alert_active:
                self.alert_active = True
                if self.alert_sound: self.alert_sound.play(loops=-1)
            elif not harmful_object_detected and self.alert_active:
                self.alert_active = False
                if self.alert_sound: self.alert_sound.stop()

            frame_count += 1
            if time.time() - start_time > 1:
                fps = frame_count / (time.time() - start_time)
                start_time = time.time()
                frame_count = 0

            fps_text = f"FPS: {fps:.2f}"
            cv2.putText(frame_resized, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            try:
                self.frame_queue.put_nowait(frame_resized)
            except queue.Full:
                pass

        if self.cap: self.cap.release()
        print("Detection thread finished.")

    def update_frame(self):
        """(UI Thread) Gets processed frames from the detection queue and displays them."""
        if not self.is_detection_running:
            return

        try:
            frame = self.frame_queue.get_nowait()
            new_h, new_w = frame.shape[:2]
            scale = min(self.screen_width / new_w, self.screen_height / new_h)
            display_w, display_h = int(new_w * scale), int(new_h * scale)
            frame_display = cv2.resize(frame, (display_w, display_h))

            display_img = Image.new('RGB', (self.screen_width, self.screen_height), 'black')
            frame_rgb = cv2.cvtColor(frame_display, cv2.COLOR_BGR2RGB)
            img_pil = Image.fromarray(frame_rgb)

            paste_x = (self.screen_width - display_w) // 2
            paste_y = (self.screen_height - display_h) // 2
            display_img.paste(img_pil, (paste_x, paste_y))

            imgtk = ImageTk.PhotoImage(image=display_img)
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)

        except queue.Empty:
            pass

        self.root.after(10, self.update_frame)

    # --------------------------------------------------------------------------
    # 8. CLEANUP
    # --------------------------------------------------------------------------

    def _stop_all_animations(self):
        """Safely stops all running Tkinter animations."""
        if self.bg_animation_job: self.root.after_cancel(self.bg_animation_job)
        if self.pulse_animation_job: self.root.after_cancel(self.pulse_animation_job)
        if self.title_animation_job: self.root.after_cancel(self.title_animation_job)
        self.is_bg_video_running = False
        if self.bg_video_thread and self.bg_video_thread.is_alive():
            self.bg_video_thread.join(timeout=1)

    def on_closing(self):
        """Handles the application window being closed."""
        print("Closing application...")
        self.is_detection_running = False
        self._stop_all_animations()

        if self.detection_thread and self.detection_thread.is_alive():
            self.detection_thread.join(timeout=1)

        if self.cap: self.cap.release()
        if self.bg_video_cap: self.bg_video_cap.release()

        pygame.mixer.quit()
        self.root.destroy()


# ==============================================================================
# 9. MAIN EXECUTION BLOCK
# ==============================================================================
def main():
    """Initializes and runs the Tkinter application."""
    root = tk.Tk()
    app = ObjectDetectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

