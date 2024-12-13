import pandas as pd
from scipy.io import arff

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    PowerTransformer,
)
from classifyAI.config.enums import NORMALISATIONS


class PreProcessing:
    def __init__(self) -> None:
        self.df = pd.DataFrame()
        self.df_scaled = pd.DataFrame()

    def load_file(self, filepath):
        absolute_path = "/home/hakou/Desktop/BioInfo/Data-mining/Data"
        data, _ = arff.loadarff(f"{absolute_path}/{filepath}")
        self.df = pd.DataFrame(data)

    def transform_into_numeric_value(self):
        le = LabelEncoder()
        self.df["class"] = le.fit_transform(self.df["class"])

    def scale_features(self, scale_method):
        match scale_method:
            case NORMALISATIONS.Z_SCORE.value:
                scaler = StandardScaler()
            case NORMALISATIONS.MIN_MAX.value:
                scaler = MinMaxScaler()
            case NORMALISATIONS.POWER.value:
                scaler = PowerTransformer()

            case _:
                scaler = StandardScaler()

        X = self.df.drop("class", axis=1)
        y = self.df["class"]

        self.df_scaled = pd.DataFrame(
            scaler.fit_transform(X), columns=X.columns, index=self.df.index
        )
        self.df_scaled["class"] = y
