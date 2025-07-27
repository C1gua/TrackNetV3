import tkinter as tk
from tkinter import filedialog
import threading
import time

import cv2
import numpy as np
import torch
import psutil
from collections import deque
from PIL import Image

from predict import predict
from utils.general import get_model, HEIGHT, WIDTH


def compute_median(cap, num_frames):
    frames = []
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    if len(frames) == 0:
        return None, frames
    median = np.median(np.array(frames), 0).astype('uint8')[..., ::-1]
    return median, frames


def preprocess_frames(frames, bg_mode='', median=None):
    processed = np.array([]).reshape(0, HEIGHT, WIDTH)
    for f in frames:
        img = Image.fromarray(f[..., ::-1])
        if bg_mode == 'subtract':
            diff = Image.fromarray(np.sum(np.absolute(np.array(img) - median), 2).astype('uint8'))
            diff = np.array(diff.resize((WIDTH, HEIGHT)))
            diff = diff.reshape(1, HEIGHT, WIDTH)
            img_arr = diff
        elif bg_mode == 'subtract_concat':
            diff = Image.fromarray(np.sum(np.absolute(np.array(img) - median), 2).astype('uint8'))
            diff = np.array(diff.resize((WIDTH, HEIGHT)))
            diff = diff.reshape(1, HEIGHT, WIDTH)
            img = np.array(img.resize((WIDTH, HEIGHT)))
            img = np.moveaxis(img, -1, 0)
            img_arr = np.concatenate((img, diff), axis=0)
        else:
            img = np.array(img.resize((WIDTH, HEIGHT)))
            img_arr = np.moveaxis(img, -1, 0)
        processed = np.concatenate((processed, img_arr), axis=0)
    if bg_mode == 'concat' and median is not None:
        processed = np.concatenate((median, processed), axis=0)
    processed = processed / 255.
    return processed


class RealtimeGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title('TrackNetV3 Realtime')
        self.video_path = None
        self.model_path = tk.StringVar()
        self._build_widgets()

    def _build_widgets(self):
        btn_open = tk.Button(self.root, text='Open Video', command=self.open_video)
        btn_open.pack(pady=2)
        tk.Entry(self.root, textvariable=self.model_path, width=60).pack(pady=2)
        btn_start = tk.Button(self.root, text='Start', command=self.start_predict)
        btn_start.pack(pady=2)
        self.fps_label = tk.Label(self.root, text='FPS: 0.00')
        self.fps_label.pack()
        self.cpu_label = tk.Label(self.root, text='CPU: 0%')
        self.cpu_label.pack()
        self.mem_label = tk.Label(self.root, text='MEM: 0%')
        self.mem_label.pack()

    def open_video(self):
        path = filedialog.askopenfilename()
        if path:
            self.video_path = path

    def start_predict(self):
        if not self.model_path.get():
            tk.messagebox.showerror('Error', 'Select TrackNet checkpoint file')
            return
        source = self.video_path if self.video_path else 0
        threading.Thread(target=self.run_predict, args=(source,), daemon=True).start()

    def update_stats(self, fps, cpu, mem):
        self.fps_label.config(text=f'FPS: {fps:.2f}')
        self.cpu_label.config(text=f'CPU: {cpu:.1f}%')
        self.mem_label.config(text=f'MEM: {mem:.1f}%')

    def run_predict(self, source):
        tracknet_ckpt = torch.load(self.model_path.get())
        seq_len = tracknet_ckpt['param_dict']['seq_len']
        bg_mode = tracknet_ckpt['param_dict']['bg_mode']
        tracknet = get_model('TrackNet', seq_len, bg_mode).cuda()
        tracknet.load_state_dict(tracknet_ckpt['model'])
        tracknet.eval()

        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print('Failed to open video source.')
            return

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w_scaler, h_scaler = w / WIDTH, h / HEIGHT
        img_scaler = (w_scaler, h_scaler)

        median_img = None
        frame_buffer = deque(maxlen=seq_len)
        if bg_mode:
            median_img, first_frames = compute_median(cap, 60)
            for f in first_frames[-seq_len:]:
                frame_buffer.append(f)
            if bg_mode == 'concat' and median_img is not None:
                img = Image.fromarray(median_img.astype('uint8'))
                img = np.array(img.resize((WIDTH, HEIGHT)))
                median_img = np.moveaxis(img, -1, 0)

        prev_pos = None
        timer = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_buffer.append(frame)
            disp_frame = frame.copy()
            color = (0, 255, 255)
            if len(frame_buffer) == seq_len:
                frames = preprocess_frames(list(frame_buffer), bg_mode, median_img)
                x = torch.from_numpy(frames).unsqueeze(0).float().cuda()
                with torch.no_grad():
                    y_pred = tracknet(x).cpu()
                indices = torch.tensor([[(0, i) for i in range(seq_len)]])
                pred = predict(indices, y_pred=y_pred, img_scaler=img_scaler)
                x_pred = pred['X'][-1]
                y_pred_c = pred['Y'][-1]
                vis = pred['Visibility'][-1]
                if vis:
                    if prev_pos is not None:
                        speed = np.linalg.norm(np.array([x_pred, y_pred_c]) - prev_pos)
                        if speed > 10:
                            color = (0, 0, 255)
                    prev_pos = np.array([x_pred, y_pred_c])
                    cv2.circle(disp_frame, (x_pred, y_pred_c), 5, color, -1)
            now = time.time()
            fps_disp = 1.0 / (now - timer) if now > timer else 0.0
            timer = now
            cpu = psutil.cpu_percent(interval=0)
            mem = psutil.virtual_memory().percent
            self.root.after(0, self.update_stats, fps_disp, cpu, mem)
            cv2.imshow('Realtime Prediction', disp_frame)
            if cv2.waitKey(1 if fps==0 else int(1000 / fps)) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

    def run(self):
        self.root.mainloop()


if __name__ == '__main__':
    gui = RealtimeGUI()
    gui.run()

