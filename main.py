import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from ttkthemes import ThemedTk

from classifyAI.common.model_predictions import ModelPredictions
from classifyAI.common.preprocessing import PreProcessing
from classifyAI.config.enums import ALGORITHMS, NORMALISATIONS
from classifyAI.widgets import combobox
from classifyAI.widgets.file_input import FileInput


class Encryptor:
    def __init__(self) -> None:
        self.root = ThemedTk(theme="arc")
        self.root.title("ClassifyAI APP")
        self.root.resizable(False, False)
        self.root.maxsize(1080, 720)

        self.current_normalisation = tk.StringVar()
        self.current_missing_val = tk.StringVar()
        self.current_algorithm = tk.StringVar()
        self.current_dataset = tk.StringVar()
        self.test_entry = tk.StringVar()
        self.measures = {}

        self.build()

    def build_first_row(self):
        config_frame = ttk.Frame(self.input_frame)
        config_frame.grid(row=0, column=0, sticky=tk.NSEW)

        # Import Dataset
        self.current_dataset = FileInput().open_dialog(config_frame)

        # Choose supervised algorithm
        self.current_algorithm = combobox.combo(
            config_frame,
            "Choose algorithm",
            (
                ALGORITHMS.KNN.value,
                ALGORITHMS.NAIVE_BAYES.value,
                ALGORITHMS.DECISION_TREE.value,
            ),
            col=1,
        )

        # Test percentage
        ttk.Label(config_frame, text="Test percentage (%)").grid(
            row=2, column=0, padx=10, sticky=tk.SW
        )
        key_entry = ttk.Entry(config_frame, textvariable=self.test_entry, width=30)
        key_entry.grid(row=3, column=0, padx=10, pady=10, sticky=tk.SW)

        # Normalisation
        self.current_normalisation = combobox.combo(
            config_frame,
            "Normalisation",
            (
                NORMALISATIONS.Z_SCORE.value,
                NORMALISATIONS.MIN_MAX.value,
                NORMALISATIONS.POWER.value,
            ),
            row=2,
            col=1,
        )

    def build_second_row(self):
        config_frame = ttk.Frame(self.input_frame)
        config_frame.grid(row=1, column=0, sticky=tk.NSEW)

        # test method
        test_button = ttk.Button(
            config_frame, text="Test model", command=self.test_model, width=50
        )
        test_button.grid(row=1, column=1, padx=10, pady=10, columnspan=2, sticky=tk.EW)

    def build_output_frame(self):
        config_frame = ttk.Frame(self.output_frame)
        config_frame.grid(row=0, column=1, sticky=tk.W, rowspan=2)

        # performance
        ttk.Label(config_frame, text="Performance measure").grid(row=0, column=1)
        self.performance_msg = ScrolledText(
            config_frame, wrap=tk.WORD, width=40, height=8
        )
        self.performance_msg.grid(row=1, column=1, padx=10, pady=10)

        # PREDICTION
        ttk.Label(config_frame, text="Prediction").grid(row=2, column=1)
        self.prediction_msg = ScrolledText(
            config_frame, wrap=tk.WORD, width=40, height=6
        )
        self.prediction_msg.grid(row=3, column=1, padx=10, pady=10)

    def build(self):
        self.input_frame = ttk.Frame(self.root).grid(row=0, column=0)
        self.output_frame = ttk.Frame(self.root).grid(row=0, column=1)

        self.build_first_row()
        self.build_second_row()

        self.build_output_frame()

    def test_model(self):
        # Pre processing phase
        pre_processing = PreProcessing()
        pre_processing.load_file(self.current_dataset.get())
        pre_processing.transform_into_numeric_value()
        pre_processing.scale_features(self.current_normalisation.get())

        # Predictions phase
        model_predictions = ModelPredictions(pre_processing.df_scaled)
        model_predictions.split_dataset(int(self.test_entry.get()) / 100)

        match self.current_algorithm.get():
            case ALGORITHMS.KNN.value:
                self.measures = model_predictions.knn()
            case ALGORITHMS.NAIVE_BAYES.value:
                self.measures = model_predictions.naive_bayes()
            case ALGORITHMS.DECISION_TREE.value:
                self.measures = model_predictions.decision_tree()

        self.performance_msg.delete("1.0", "end")
        for k, v in self.measures.items():
            self.performance_msg.insert("1.0", f"{k}: {v}\n")


if __name__ == "__main__":
    app = Encryptor()
    app.root.mainloop()
