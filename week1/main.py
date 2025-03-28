import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from src.utils import load_dataset
from src.id3 import build_tree, predict


def main():
    dataset_path = os.path.join("data", "diabetes_dataset.csv")
    data = load_dataset(dataset_path)
    target = "diabet"
    train_size = int(0.8 * len(data))
    train_data = data[:train_size]
    test_data = data[train_size:]
    tree = build_tree(train_data, target)
    y_true = test_data[target].values
    y_pred = [predict(tree, row) for _, row in test_data.iterrows()]
    print("Acuratețea modelului:", accuracy_score(y_true, y_pred))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title("Matricea de Confuzie")
    plt.savefig("confusion_matrix.png")


if __name__ == "__main__":
    main()
