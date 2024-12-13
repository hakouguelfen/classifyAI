from tkinter import ttk, filedialog
import tkinter as tk


class FileInput:
    def __init__(self) -> None:
        self.dataset_entry = ttk.Entry()

    def select_file(self, _):
        file_path = filedialog.askopenfilename(
            title="Select a file",
            filetypes=[("Text files", "*.arff"), ("All files", "*.*")],
        )
        if file_path:
            file_path = "".join(file_path.split("/")[-1:])
            self.dataset_entry.insert("0", file_path)

    def open_dialog(self, frame, row=0, col=0):
        text_var = tk.StringVar()

        ttk.Label(frame, text="Import file").grid(
            row=row, column=col, padx=10, sticky=tk.SW
        )

        self.dataset_entry = ttk.Entry(frame, textvariable=text_var, width=30)
        self.dataset_entry.bind("<Button-1>", self.select_file)
        self.dataset_entry.grid(row=row + 1, column=col, padx=10, pady=10, sticky=tk.SW)

        return text_var
