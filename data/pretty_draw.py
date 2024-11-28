import tkinter as tk
from tkinter import simpledialog
import json
import uuid
import time

from constants import PRETTY_COORD_FOLDER

class PrettyDrawingApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Pretty Drawing App")

        self.canvas = tk.Canvas(master, width=800, height=800, bg="white")
        self.canvas.pack()

        self.save_button = tk.Button(master, text="Save", command=self.save)
        self.save_button.pack(side=tk.LEFT)

        self.clear_button = tk.Button(master, text="Clear", command=self.clear)
        self.clear_button.pack(side=tk.RIGHT)

        self.pretty_button = tk.Button(master, text="Draw Pretty Version", command=self.start_pretty_drawing)
        self.pretty_button.pack(side=tk.BOTTOM)

        self.original_tokens = []
        self.pretty_tokens = []
        self.current_tokens = self.original_tokens
        self.drawing = False
        self.last_time = 0
        self.pen_lifted = False
        self.is_pretty_mode = False

        self.canvas.bind("<Button-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

    def start_drawing(self, event):
        self.drawing = True
        self.last_time = time.time()
        if self.pen_lifted and self.current_tokens:
            self.current_tokens.append({"type": "pause"})
            self.pen_lifted = False
        self.draw(event)

    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            color = "red" if self.is_pretty_mode else "black"
            self.canvas.create_oval(x-2, y-2, x+2, y+2, fill=color)
            current_time = time.time()
            if current_time - self.last_time > 0.3:
                self.current_tokens.append({"type": "pause"})
            self.current_tokens.append({"type": "coordinate", "x": x, "y": y})
            self.last_time = current_time

    def stop_drawing(self, event):
        self.drawing = False
        self.pen_lifted = True

    def start_pretty_drawing(self):
        if not self.is_pretty_mode:
            self.is_pretty_mode = True
            self.current_tokens = self.pretty_tokens
            self.pretty_button.config(text="Finish Pretty Drawing")
        else:
            self.is_pretty_mode = False
            self.pretty_button.config(text="Draw Pretty Version")

    def save(self):
        self.original_tokens.insert(0, {"type": "start"})
        self.original_tokens.append({"type": "end"})
        self.pretty_tokens.insert(0, {"type": "start"})
        self.pretty_tokens.append({"type": "end"})
        
        word = simpledialog.askstring("Input", "Enter the word you drew:")
        if word:
            data = {
                "word": word,
                "original_tokens": self.original_tokens,
                "pretty_tokens": self.pretty_tokens
            }
            filename = f"{uuid.uuid4()}.json"
            file_path = PRETTY_COORD_FOLDER / filename
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Saved to {file_path}")

    def clear(self):
        self.canvas.delete("all")
        self.original_tokens = []
        self.pretty_tokens = []
        self.current_tokens = self.original_tokens
        self.pen_lifted = False
        self.is_pretty_mode = False
        self.pretty_button.config(text="Draw Pretty Version")

def clear_folder():
    for file in PRETTY_COORD_FOLDER.iterdir():
        file.unlink()

if __name__ == "__main__":
    #clear_folder()
    root = tk.Tk()
    app = PrettyDrawingApp(root)
    root.mainloop()