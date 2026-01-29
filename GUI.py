import cv2
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import time
import psutil
import GPUtil
from ultralytics import YOLO
import torch
import threading

MODEL = r'C:\Users\USER\Downloads\Regnum-20260129T090605Z-3-001\Regnum\runs\regnum_traffic_model\weights\best.pt'
VIDEO_IN  = r'C:\Users\USER\Downloads\Regnum-20260129T090605Z-3-001\Regnum\Video\Supporting video for Dataset-2.mp4'
VIDEO_OUT = r'C:\Users\USER\Downloads\Regnum-20260129T090605Z-3-001\Regnum\Video\processed_video.mp4'

model = YOLO(MODEL).to('cuda' if torch.cuda.is_available() else 'cpu')


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Traffic Monitor")

        self.c = tk.Canvas(root, width=1280, height=720)
        self.c.pack()

        f = ttk.Frame(root)
        f.pack()

        ttk.Button(f, text="Start", command=self.start).pack(side='left')
        ttk.Button(f, text="Stop",  command=self.stop).pack(side='left')
        ttk.Button(f, text="Process Video", command=self.process_video).pack(side='left')

        mf = ttk.Frame(root)
        mf.pack()

        self.lb_fps  = ttk.Label(mf, text="FPS: —");   self.lb_fps.pack(side='left')
        self.lb_cpu  = ttk.Label(mf, text="CPU: —");   self.lb_cpu.pack(side='left')
        self.lb_gpu  = ttk.Label(mf, text="GPU: —");   self.lb_gpu.pack(side='left')
        self.lb_objs = ttk.Label(mf, text="Objs: —");  self.lb_objs.pack(side='left')

        self.cap = None
        self.run = False
        self.t = None

        self.fps = 0
        self.t0  = 0
        self.n   = 0

    def start(self):
        if self.run: return
        self.cap = cv2.VideoCapture(VIDEO_IN)
        self.run = True
        self.t = threading.Thread(target=self.loop, daemon=True)
        self.t.start()

    def stop(self):
        self.run = False
        if self.cap:
            self.cap.release()
            self.cap = None

    def loop(self):
        while self.run:
            ret, frame = self.cap.read()
            if not ret:
                self.stop()
                break

            res = model.track(frame, persist=True)[0]
            self.n = len(res.boxes)

            frame = self.draw(res, frame)
            self.update_stats(res)
            self.show(frame)

            time.sleep(0.005)   # gentle on CPU

        if self.cap:
            self.cap.release()

    def draw(self, r, frame):
        for box in r.boxes:
            if box.id is None: continue

            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cls = int(box.cls)
            tid = int(box.id)
            conf = box.conf.item()

            label = f"{r.names[cls]} {tid} {conf:.2f}"

            cv2.rectangle(frame, (x1,y1), (x2,y2), (0,240,60), 2)
            cv2.putText(frame, label, (x1,y1-8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0,240,60), 2)

        return frame

    def update_stats(self, r):
        t = time.time()
        self.fps = 1 / (t - self.t0) if self.t0 > 0 else 0
        self.t0 = t

        cpu = psutil.cpu_percent(interval=None)
        gpu = GPUtil.getGPUs()[0].load * 100 if GPUtil.getGPUs() else 0

        self.root.after(0, self.lb_fps .config, {'text': f"FPS: {self.fps:4.1f}"})
        self.root.after(0, self.lb_cpu .config, {'text': f"CPU: {cpu:4.1f}%"})
        self.root.after(0, self.lb_gpu .config, {'text': f"GPU: {gpu:4.1f}%"})
        self.root.after(0, self.lb_objs.config, {'text': f"Objs: {self.n:3d}"})

    def show(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (1280, 720))
        img = ImageTk.PhotoImage(Image.fromarray(frame))
        self.c.img = img               # keep reference!
        self.c.create_image(0, 0, anchor='nw', image=img)

    def process_video(self):
        cap = cv2.VideoCapture(VIDEO_IN)
        if not cap.isOpened():
            print("Cannot open input video")
            return

        w = int(cap.get(3))
        h = int(cap.get(4))
        out = cv2.VideoWriter(VIDEO_OUT, cv2.VideoWriter_fourcc(*'mp4v'), 30, (w,h))

        print("Processing video...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break

            res = model.track(frame, persist=True)[0]
            frame = self.draw(res, frame)
            out.write(frame)

        cap.release()
        out.release()
        print(f"Saved → {VIDEO_OUT}")


if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()