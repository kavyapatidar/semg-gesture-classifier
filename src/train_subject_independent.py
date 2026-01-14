import numpy as np
import gc

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from src.features import extract_features_from_windows


def subject_independent_evaluation(subject_data):
    """
    Leave-One-Subject-Out evaluation (ultra memory-safe version)
    """

    subject_ids = list(subject_data.keys())
    accuracies = []

    for test_subject in subject_ids:
        print(f"\nTesting on unseen subject: {test_subject}")

        # Collect train/test
        X_train_list = []
        y_train_list = []

        X_test = subject_data[test_subject]["X"]
        y_test = subject_data[test_subject]["y"]

        for subject_id in subject_ids:
            if subject_id == test_subject:
                continue

            X_train_list.append(subject_data[subject_id]["X"])
            y_train_list.append(subject_data[subject_id]["y"])

        X_train = np.concatenate(X_train_list, axis=0)
        y_train = np.concatenate(y_train_list, axis=0)

        # Extract features (FLOAT32)
        X_train_feat = extract_features_from_windows(X_train).astype(np.float32)
        X_test_feat = extract_features_from_windows(X_test).astype(np.float32)

        # Free raw windows immediately
        del X_train, X_test
        gc.collect()

        # Scale
        scaler = StandardScaler()
        X_train_feat = scaler.fit_transform(X_train_feat)
        X_test_feat = scaler.transform(X_test_feat)

        # VERY MEMORY-SAFE MODEL
        model = RandomForestClassifier(
            n_estimators=50,     # further reduced
            max_depth=15,
            min_samples_leaf=5,
            n_jobs=1,
            random_state=42
        )

        model.fit(X_train_feat, y_train)
        y_pred = model.predict(X_test_feat)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        print(f"Accuracy for {test_subject}: {acc:.4f}")

        # Explicit cleanup
        del X_train_feat, X_test_feat, model
        gc.collect()

    print("\n==============================")
    print("Subject-Independent Results")
    print("==============================")
    print(f"Mean Accuracy: {np.mean(accuracies):.4f}")
    print(f"Std Deviation: {np.std(accuracies):.4f}")

    return accuracies
