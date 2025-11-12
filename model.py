import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
import argparse
from pathlib import Path


# random seed
seed = 1


def train_and_save(out_path: str = "rf_model.sav", compress: int | bool = 3) -> float:
    """Train a RandomForest on the Iris dataset and save the model.

    Returns the accuracy on the held-out test set.
    """
    # Read original dataset
    iris = load_iris()
    X = iris.data
    y = iris.target

    # split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    # create an instance of the random forest classifier
    clf = RandomForestClassifier(n_estimators=100, random_state=seed)

    # train the classifier on the training data
    clf.fit(X_train, y_train)

    # predict on the test set
    y_pred = clf.predict(X_test)

    # calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    # ensure output dir exists
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    # save the model to disk (joblib compress for smaller file)
    joblib.dump(clf, out_path, compress=compress)

    return float(accuracy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and save Iris RandomForest model")
    parser.add_argument("--out", "-o", default="rf_model.sav", help="Output model path")
    parser.add_argument("--no-compress", dest="compress", action="store_false", help="Disable joblib compression")
    parser.add_argument("--compress-level", type=int, default=3, help="joblib compress level (int) or 0/False to disable")
    args = parser.parse_args()

    compress = args.compress_level if args.compress else False
    acc = train_and_save(out_path=args.out, compress=compress)
    print(f"Trained RandomForest and saved to {args.out}")
    print(f"Accuracy on test set: {acc:.3f}")


if __name__ == "__main__":
    main()
