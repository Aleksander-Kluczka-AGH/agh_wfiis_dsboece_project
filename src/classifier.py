import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


def classifier(df):
    # Step 2: Load and preprocess the labeled data
    # Assuming you have a DataFrame df with 'RSRP', 'RSRQ', and 'label' columns

    X = df[["rsrp", "rsrq"]].values
    y = df["label"].values

    # Normalize the input features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Step 3: Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Step 4: Define the MLP model
    mlp = MLPClassifier(
        hidden_layer_sizes=(100, 50), activation="relu", max_iter=1500, random_state=42
    )

    # Step 5: Train the model on the training data
    history = mlp.fit(X_train, y_train)

    # Plot the loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(
        np.arange(1, len(history.loss_curve_) + 1), history.loss_curve_, marker="o"
    )
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.show()

    # Step 6: Evaluate the model on the testing data
    # Step 6: Evaluate the model on the testing data
    y_pred = mlp.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    print("Accuracy:", accuracy)
    print("Confusion matrix:\n", cm)

    # TODO: Make it sorted!
    class_labels = df["label"].unique().tolist()

    # Create a heatmap of the confusion matrix
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
