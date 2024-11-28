import tkinter as tk
from tkinter import simpledialog
import json
import uuid
import time

from constants import RAW_COORD_FOLDER

class DrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Drawing App")

        self.canvas = tk.Canvas(master, width=800, height=800, bg="white")
        self.canvas.pack()

        self.save_button = tk.Button(master, text="Save", command=self.save)
        self.save_button.pack(side=tk.LEFT)

        self.clear_button = tk.Button(master, text="Clear", command=self.clear)
        self.clear_button.pack(side=tk.RIGHT)

        self.coordinates = []
        self.drawing = False
        self.last_time = 0
        self.pen_lifted = False

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

    def start_drawing(self, event):
        self.drawing = True
        self.last_time = time.time()
        if self.pen_lifted:
            self.coordinates.append({"type": "pause"})
            self.pen_lifted = False
        self.draw(event)

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            self.canvas.create_oval(x-2, y-2, x+2, y+2, fill="black")
            current_time = time.time()
            if current_time - self.last_time > 0.3:
                self.coordinates.append({"type": "pause"})
            self.coordinates.append({"type": "coordinate", "x": x, "y": y})
            self.last_time = current_time

    def stop_drawing(self, event):
        self.drawing = False
        self.pen_lifted = True

    def save(self):
        self.coordinates.insert(0, {"type": "start"})
        self.coordinates.append({"type": "end"})
        word = simpledialog.askstring("Input", "Enter the word you drew:")
        if word:
            data = {
                "word": word,
                "tokens": self.coordinates
            }
            filename = f"{uuid.uuid4()}.json"
            file_path = RAW_COORD_FOLDER / filename
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved to {file_path}")

    def clear(self):
        self.canvas.delete("all")
        self.coordinates = []
        self.pen_lifted = False

def clear_folder():
    for file in RAW_COORD_FOLDER.iterdir():
        file.unlink()

if __name__ == "__main__":
    #clear_folder()
    root = tk.Tk()
    app = DrawingApp(root)
    root.mainloop()