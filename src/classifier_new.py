from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pymoo.core.problem import ElementwiseProblem
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    import pandas as pd


# bounds of the RSRP and RSRQ
# rsrp_th = [-140, -44]
# rsrq_th = [-19.5, -3]


class Classifier(ElementwiseProblem):
    def __init__(self, df: "pd.DataFrame"):
        super().__init__(
            n_var=2,
            n_obj=2,
            n_ieq_constr=2,
            xl=np.array([-140, -19.5]),  # lower bounds
            xu=np.array([-44, -3]),  # upper bounds
        )

        self.df: "pd.DataFrame" = df
        X = df[["rsrp", "rsrq", "pci", "rssi", "ta", "mnc"]].values
        y = df["label"].values

        X = StandardScaler().fit_transform(X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.mlp = MLPClassifier(
            hidden_layer_sizes=(400, 100),
            activation="logistic",
            solver="sgd",
            max_iter=2500,
            random_state=42,
            learning_rate="adaptive",
            learning_rate_init=0.1,
        )

    def train(self) -> MLPClassifier:
        self.history = self.mlp.fit(self.X_train, self.y_train)
        return self.history

    def plot_loss_curve(self):
        plt.figure(figsize=(8, 6))
        plt.plot(
            np.arange(1, len(self.history.loss_curve_) + 1),
            self.history.loss_curve_,
            marker="o",
        )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curve")
        plt.show()

    def test(self):
        self.y_pred = self.mlp.predict(self.X_test)

    def confusion_matrix(self):
        accuracy = accuracy_score(self.y_test, self.y_pred)
        cm = confusion_matrix(self.y_test, self.y_pred)
        print("Accuracy:", accuracy)
        print("Confusion matrix:\n", cm)

        class_labels = self.df["label"].unique().tolist()

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_labels,
            yticklabels=class_labels,
        )
        plt.xlabel("Predicted Labels")
        plt.ylabel("True Labels")
        plt.title("Confusion Matrix")
        plt.show()

    def _evaluate(self, x, out, *args, **kwargs):
        """
        The function's interface is the parameters x and out. For this element-wise
        implementation x is a one-dimensional NumPy array of length n_var which represents a single
        solution to be evaluated. The output is supposed to be written to the dictionary out.

        The objective values should be written to out["F"] as a list of NumPy array with length of
        n_obj and the constraints to out["G"] with length of n_constr (if the problem has
        constraints to be satisfied at all).

        """
        # f1 = 100 * (x[0]**2 + x[1]**2)
        # f2 = (x[0]-1)**2 + x[1]**2

        # g1 = 2*(x[0]-0.1) * (x[0]-0.9) / 0.18
        # g2 = - 20*(x[0]-0.4) * (x[0]-0.6) / 4.8

        # out["F"] = [f1, f2]
        # out["G"] = [g1, g2]
        # f =
        g = self.mlp.predict(x)
        out["G"] = [g]
