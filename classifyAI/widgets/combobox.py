from tkinter import ttk
import tkinter as tk


def combo(frame, label, values, row=0, col=0) -> tk.StringVar:
    text_var = tk.StringVar()

    ttk.Label(frame, text=label).grid(row=row, column=col, padx=10, sticky=tk.SW)
    combobox_entry = ttk.Combobox(
        frame,
        textvariable=text_var,
        width=30,
        state="readonly",
    )
    combobox_entry["value"] = values
    combobox_entry.current(0)
    combobox_entry.grid(row=row + 1, column=col, padx=10, pady=10, sticky=tk.SW)

    return text_var
