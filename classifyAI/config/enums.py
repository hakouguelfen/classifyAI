from enum import Enum


class ALGORITHMS(Enum):
    KNN = "KNN"
    NAIVE_BAYES = "Naive Bayes"
    DECISION_TREE = "Decision Tree"


class NORMALISATIONS(Enum):
    Z_SCORE = "z_score"
    MIN_MAX = "min max"
    POWER = "power transform"


# class MISSINGVALUES(Enum):
#     MEAN = "mean"
#     MEDIAN = "median"
#     MODE = "mode"
#     MIN = "min"
#     MAX = "max"
#     VARIANCE = "var"
#     STANDARD_DEVIATION = "std"
