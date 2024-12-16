from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


import joblib


class ModelPredictions:
    def __init__(self, df_scaled) -> None:
        self.knn = None
        self.gaussianNB = None
        self.decision_tree_classifier = None
        self.svm_model = None
        self.neural_network_model = None

        self.df_scaled = df_scaled
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

    def split_dataset(self, test_size, random_state):
        X = self.df_scaled.drop("class", axis=1)  # All columns except 'class'
        y = self.df_scaled["class"]

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

    def knn_predict(self, sample):
        if self.knn is None:
            self.k_neighbors()

        knn = joblib.load("models/knn.pkl")
        return knn.predict(sample)

    def naive_bayes_predict(self, sample):
        if self.gaussianNB is None:
            self.naive_bayes()

        gaussianNB = joblib.load("models/naive_bayes.pkl")
        return gaussianNB.predict(sample)

    def decision_tree_predict(self, sample):
        if self.decision_tree_classifier is None:
            self.decision_tree()

        decision_tree = joblib.load("models/decision_tree.pkl")
        return decision_tree.predict(sample)

    def neural_network_predict(self, sample):
        if self.neural_network_model is None:
            self.neural_network()

        neural_network = joblib.load("models/neural_network.pkl")
        return neural_network.predict(sample)

    def svm_predict(self, sample):
        if self.svm_model is None:
            self.svm()

        svm = joblib.load("models/svm.pkl")
        return svm.predict(sample)

    # -----------------------------------------------------
    def k_neighbors(self):
        k = 91

        self.knn = KNeighborsClassifier(n_neighbors=k)
        self.knn.fit(self.X_train, self.y_train)

        joblib.dump(self.knn, "models/knn.pkl")

        predictions = self.knn.predict(self.X_test)
        measures = self.calculate_measures(predictions)

        return measures

    def naive_bayes(self):
        self.gaussianNB = GaussianNB()
        self.gaussianNB.fit(self.X_train, self.y_train)

        joblib.dump(self.gaussianNB, "models/naive_bayes.pkl")

        predictions = self.gaussianNB.predict(self.X_test)
        measures = self.calculate_measures(predictions)

        return measures

    def decision_tree(self):
        self.decision_tree_classifier = DecisionTreeClassifier(
            max_depth=5, max_features="sqrt", min_samples_leaf=4, random_state=147
        )
        self.decision_tree_classifier.fit(self.X_train, self.y_train)

        joblib.dump(self.decision_tree_classifier, "models/decision_tree.pkl")

        predictions = self.decision_tree_classifier.predict(self.X_test)
        measures = self.calculate_measures(predictions)

        return measures

    def neural_network(self):
        self.neural_network_model = MLPClassifier(
            hidden_layer_sizes=(120,),
            activation="relu",
            solver="adam",
            max_iter=200,
            random_state=22,
            verbose=False,
        )

        self.neural_network_model.fit(self.X_train, self.y_train)

        joblib.dump(self.neural_network_model, "models/neural_network.pkl")

        predictions = self.neural_network_model.predict(self.X_test)
        measures = self.calculate_measures(predictions)
        return measures

    def svm(self):
        self.svm_model = SVC(kernel="linear")
        self.svm_model.fit(self.X_train, self.y_train)

        joblib.dump(self.svm_model, "models/svm.pkl")

        predictions = self.svm_model.predict(self.X_test)
        measures = self.calculate_measures(predictions)

        return measures
