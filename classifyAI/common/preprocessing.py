import pandas as pd
import numpy as np
from scipy.io import arff

from sklearn.preprocessing import (
    StandardScaler,
    RobustScaler,
    LabelEncoder,
)


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

    def scale_features(self):
        std_scaler = StandardScaler()
        robust_scaler = RobustScaler()

        self.df_scaled = self.df.copy()

        for var in ["insu", "pedi"]:
            self.df_scaled[var] = np.log1p(self.df[var])

        normal_vars = ["plas", "pres"]
        self.df_scaled[normal_vars] = std_scaler.fit_transform(self.df[normal_vars])

        robust_vars = ["preg", "skin", "mass", "age"]
        self.df_scaled[robust_vars] = robust_scaler.fit_transform(self.df[robust_vars])
