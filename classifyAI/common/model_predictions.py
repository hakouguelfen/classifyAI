from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


class ModelPredictions:
    def __init__(self, df_scaled) -> None:
        self.df_scaled = df_scaled
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

    def split_dataset(self, test_size):
        X = self.df_scaled.drop("class", axis=1)  # All columns except 'class'
        y = self.df_scaled["class"]

        random_state = 121

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=y,  # Maintain the same class distribution in train and test sets
        )

    def calculate_measures(self, predictions):
        TP = FP = TN = FN = 0

        for test, pred in zip(self.y_test, predictions):
            TP += test == 1 and pred == 1
            FP += test == 0 and pred == 1
            TN += test == 0 and pred == 0
            FN += test == 1 and pred == 0

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        rappel = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1_score = (
            2 * (precision * rappel) / (precision + rappel)
            if (precision + rappel) > 0
            else 0
        )

        return {
            "precision": round(precision, 2),
            "rappel": round(rappel, 2),
            "f1_score": round(f1_score, 2),
            "TP": TP,
            "FP": FP,
            "TN": TN,
            "FN": FN,
        }

    def knn(self):
        k = 91

        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(self.X_train, self.y_train)
        predictions = knn.predict(self.X_test)

        measures = self.calculate_measures(predictions)
        return measures

    def naive_bayes(self):
        model = GaussianNB()
        model.fit(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)

        measures = self.calculate_measures(predictions)
        return measures

    def decision_tree(self):
        clf = DecisionTreeClassifier(criterion="gini", random_state=20)
        clf.fit(self.X_train, self.y_train)
        predictions = clf.predict(self.X_test)

        measures = self.calculate_measures(predictions)
        return measures
