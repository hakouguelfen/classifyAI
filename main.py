import tkinter as tk
from tkinter import ttk
from tkinter.scrolledtext import ScrolledText
from ttkthemes import ThemedTk

from classifyAI.common.model_predictions import ModelPredictions
from classifyAI.common.preprocessing import PreProcessing
from classifyAI.config.enums import ALGORITHMS
from classifyAI.widgets import combobox
from classifyAI.widgets.file_input import FileInput


class Encryptor:
    def __init__(self) -> None:
        self.root = ThemedTk(theme="arc")
        self.root.title("ClassifyAI APP")
        self.root.resizable(False, False)
        self.root.maxsize(1080, 720)

        self.pre_processing = PreProcessing()

        self.model_predictions = None
        self.current_normalisation = tk.StringVar()
        self.current_missing_val = tk.StringVar()
        self.current_algorithm = tk.StringVar()
        self.current_dataset = tk.StringVar()
        self.test_entry = tk.StringVar()
        self.random_state_entry = tk.StringVar()
        self.predict_entry = tk.StringVar()
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
                ALGORITHMS.NEURAL_NETWORK.value,
                ALGORITHMS.SVM.value,
            ),
            col=1,
        )

        # Test percentage
        ttk.Label(config_frame, text="Test percentage (%)").grid(
            row=2, column=0, padx=10, sticky=tk.SW
        )
        key_entry = ttk.Entry(config_frame, textvariable=self.test_entry, width=30)
        key_entry.grid(row=3, column=0, padx=10, pady=10, sticky=tk.SW)

        # Random state
        ttk.Label(config_frame, text="Random state").grid(
            row=2, column=1, padx=10, sticky=tk.SW
        )
        key_entry = ttk.Entry(
            config_frame, textvariable=self.random_state_entry, width=31
        )
        key_entry.grid(row=3, column=1, padx=10, pady=10, sticky=tk.SW)

        # test method
        test_button = ttk.Button(
            config_frame, text="Test model", command=self.test_model
        )
        test_button.grid(row=4, column=1, padx=10, pady=10, sticky=tk.EW)

        separator = ttk.Separator(config_frame, orient="horizontal")
        separator.grid(row=5, column=0, columnspan=2, sticky="ew", padx=10, pady=10)

    def build_second_row(self):
        config_frame = ttk.Frame(self.input_frame)
        config_frame.grid(row=1, column=0, sticky=tk.NSEW)

        # data input to predict
        ttk.Label(config_frame, text="dataset").grid(
            row=1, column=0, padx=10, sticky=tk.SW
        )
        key_entry = ttk.Entry(config_frame, textvariable=self.predict_entry, width=30)
        key_entry.grid(row=2, column=0, padx=10, pady=10)

        # test method
        test_button = ttk.Button(
            config_frame, text="Classify data", command=self.predict
        )
        test_button.grid(row=3, column=1, padx=10, pady=10, sticky=tk.EW)

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
        self.pre_processing.load_file(self.current_dataset.get())
        self.pre_processing.transform_into_numeric_value()
        self.pre_processing.scale_features()

        # Predictions phase
        self.model_predictions = ModelPredictions(self.pre_processing.df_scaled)
        self.model_predictions.split_dataset(
            int(self.test_entry.get()) / 100, int(self.random_state_entry.get())
        )

        match self.current_algorithm.get():
            case ALGORITHMS.KNN.value:
                self.measures = self.model_predictions.k_neighbors()
            case ALGORITHMS.NAIVE_BAYES.value:
                self.measures = self.model_predictions.naive_bayes()
            case ALGORITHMS.DECISION_TREE.value:
                self.measures = self.model_predictions.decision_tree()
            case ALGORITHMS.NEURAL_NETWORK.value:
                self.measures = self.model_predictions.neural_network()
            case ALGORITHMS.SVM.value:
                self.measures = self.model_predictions.svm()

        self.performance_msg.delete("1.0", "end")
        for k, v in self.measures.items():
            self.performance_msg.insert("1.0", f"{k}: {v}\n")

    def predict(self):
        if self.model_predictions is None:
            self.test_model()

        msg = self.predict_entry.get()
        sample = list(map(lambda x: float(x), msg.split(",")))
        sample_df = self.pre_processing.scale_sample(sample)

        match self.current_algorithm.get():
            case ALGORITHMS.KNN.value:
                self.prediction = self.model_predictions.knn_predict(sample_df)
            case ALGORITHMS.NAIVE_BAYES.value:
                self.prediction = self.model_predictions.naive_bayes_predict(sample_df)
            case ALGORITHMS.DECISION_TREE.value:
                self.prediction = self.model_predictions.decision_tree_predict(
                    sample_df
                )
            case ALGORITHMS.NEURAL_NETWORK.value:
                self.measures = self.model_predictions.neural_network_predict(sample_df)
            case ALGORITHMS.SVM.value:
                self.prediction = self.model_predictions.svm_predict(sample_df)

        self.prediction_msg.delete("1.0", "end")
        if self.prediction[0] == 1:
            self.prediction_msg.insert("1.0", "class: tested_positive\n")
        else:
            self.prediction_msg.insert("1.0", "class: tested_negative\n")


if __name__ == "__main__":
    app = Encryptor()
    app.root.mainloop()
