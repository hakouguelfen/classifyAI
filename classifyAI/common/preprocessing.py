import pandas as pd
import numpy as np
from scipy.io import arff
import joblib

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

        joblib.dump(std_scaler, "scalers/std_scaler.pkl")
        joblib.dump(robust_scaler, "scalers/robust_scaler.pkl")

    def scale_sample(self, sample=None):
        std_scaler = joblib.load("scalers/std_scaler.pkl")
        robust_scaler = joblib.load("scalers/robust_scaler.pkl")

        sample_df = pd.DataFrame([sample], columns=self.df_scaled.columns.drop("class"))

        # Log scaling
        sample_df["insu"] = np.log1p(sample_df["insu"])
        sample_df["pedi"] = np.log1p(sample_df["pedi"])

        # Apply saved scalers
        normal_vars = ["plas", "pres"]
        sample_df[normal_vars] = std_scaler.transform(sample_df[normal_vars])

        robust_vars = ["preg", "skin", "mass", "age"]
        sample_df[robust_vars] = robust_scaler.transform(sample_df[robust_vars])

        return sample_df
