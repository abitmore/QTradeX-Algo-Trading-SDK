import json
import tkinter as tk
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import qtradex as qx
import ttkbootstrap as ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from qtradex.common.utilities import sigfig
from qtradex.core.tune_manager import save_tune
from ttkbootstrap.constants import *

HISTORY = 100

class MouseWheelTuner:
    def __init__(self, data, wallet):
        self.data = data
        self.wallet = wallet

    def optimize(self, bot):
        self.bot = bot
        # Initialize ttkbootstrap with the 'darkly' theme
        self.root = ttk.Window(themename="darkly")
        self.root.title("Mouse Wheel Tuner")

        self.history = []
        self.future = []

        # Create canvas areas for each coefficient
        self.knob_areas = []
        self.labels = []
        for i, (key, value) in enumerate(self.bot.tune.items()):
            # Use Canvas with theme-compatible background
            knob_area = tk.Canvas(self.root, width=100, height=20, bg='#aaaaaa', highlightthickness=1, highlightbackground='#aaaaaa')
            knob_area.create_rectangle(0, 0, 100, 20, fill="#aaaaaa")
            knob_area.grid(row=i, column=1, padx=5, pady=2)
            # Use ttk.Label for themed text
            label = ttk.Label(self.root, text=f"{key}", anchor="e")
            label.grid(sticky=ttk.E, row=i, column=0, padx=5)
            label = ttk.Label(self.root, text=f"{sigfig(value, 6)}", anchor="w")
            label.grid(sticky=ttk.W, row=i, column=2, padx=5)
            self.labels.append(label)
            knob_area.bind("<Enter>", self.on_enter)
            knob_area.bind("<Leave>", self.on_leave)
            knob_area.bind("<MouseWheel>", lambda event, k=key: self.on_scroll(event, k))
            knob_area.bind("<Button-4>", lambda event, k=key: self.on_scroll(event, k, force_up=True))
            knob_area.bind("<Button-5>", lambda event, k=key: self.on_scroll(event, k, force_down=True))
            self.knob_areas.append(knob_area)

        # Update buttons with ttkbootstrap styling
        self.update_button = ttk.Button(self.root, text="Undo", command=self.undo, width=40, bootstyle=PRIMARY)
        self.update_button.grid(row=0, column=3, padx=5, pady=2)
        self.update_button = ttk.Button(self.root, text="Redo", command=self.redo, width=40, bootstyle=PRIMARY)
        self.update_button.grid(row=1, column=3, padx=5, pady=2)
        self.update_button = ttk.Button(self.root, text="Save", command=self.save, width=40, bootstyle=SUCCESS)
        self.update_button.grid(row=2, column=3, padx=5, pady=2)

        self.entry = ttk.StringVar(self.root, value="MouseWheel")
        ttk.Entry(self.root, textvariable=self.entry, width=40, bootstyle=SECONDARY).grid(row=4, column=3, padx=5, pady=5)

        # Scrollbar and Text area
        scrollbar = ttk.Scrollbar(self.root, orient=VERTICAL, bootstyle=SECONDARY)
        scrollbar.grid(row=4, column=4, rowspan=999, sticky=NS)

        self.textarea = tk.Text(self.root, height=20, width=40, yscrollcommand=scrollbar.set, bg='#2c2c2c', fg='#ffffff', insertbackground='#ffffff')
        self.textarea.grid(row=4, column=3, rowspan=999, padx=5, pady=5)
        scrollbar.config(command=self.textarea.yview)
        self.textarea.configure(state="disabled")

        self.last_test = None
        self.results = None

        # Initial plot
        self.update_plot()
        self.root.mainloop()

    def save(self):
        bot = deepcopy(self.bot)
        bot.tune = {"tune": bot.tune, "results": self.results}
        save_tune(bot, self.entry.get())
        print("saved!")

    def undo(self):
        if self.history:
            self.future.append(self.bot.tune.copy())
            self.bot.tune = self.history.pop()
            self.update_labels()
            self.update_plot()

    def redo(self):
        if self.future:
            self.history.append(self.bot.tune.copy())
            self.bot.tune = self.future.pop()
            self.update_labels()
            self.update_plot()

    def on_enter(self, event):
        event.widget.config(cursor="hand2")
        event.widget.focus_set()

    def on_leave(self, event):
        event.widget.config(cursor="")

    def on_scroll(self, event, key, force_up=False, force_down=False):
        if event.state & 0x0001:  # Shift key pressed
            increment = 0.001
        else:
            increment = 0.01

        if force_up or (event.delta > 0 and not force_down):  # Scroll up
            self.bot.tune[key] *= 1 + increment
        elif force_down or (event.delta < 0):  # Scroll down
            self.bot.tune[key] *= 1 - increment

        self.bot.tune[key] = max(0, min(100, self.bot.tune[key]))
        self.update_labels()

        if not self.history or self.history[-1] != self.bot.tune:
            self.history.append(self.bot.tune.copy())
            if len(self.history) > HISTORY:
                self.history.pop(0)

    def update_labels(self):
        for i, (key, value) in enumerate(self.bot.tune.items()):
            self.labels[i].config(text=f"{sigfig(value, 6)}")

    def update_plot(self):
        if self.bot.tune != self.last_test:
            self.last_test = self.bot.tune.copy()
            self.results = qx.backtest(self.bot, self.data, block=False, show=False)
            self.textarea.configure(state='normal')
            self.textarea.delete('1.0', tk.END)
            self.textarea.insert(tk.END, json.dumps(self.results, indent=4))
            self.textarea.configure(state='disabled')
        self.root.after(5000, self.update_plot)
