import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

def main():
    # Tentukan path absolut berdasarkan lokasi file automate.py
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(BASE_DIR, "..", "breast_raw", "data.csv")

    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Clean data
    df = df.drop(['id', 'Unnamed: 32'], axis=1, errors='ignore')
    df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    
    X = df.drop('diagnosis', axis=1)
    y = df['diagnosis']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Save processed files ke root project
    OUTPUT_TRAIN = os.path.join(BASE_DIR, "..", "breast_preprocessed_train.csv")
    OUTPUT_TEST = os.path.join(BASE_DIR, "..", "breast_preprocessed_test.csv")

    pd.DataFrame(X_train_scaled, columns=X.columns).assign(
        diagnosis=y_train.values
    ).to_csv(OUTPUT_TRAIN, index=False)

    pd.DataFrame(X_test_scaled, columns=X.columns).assign(
        diagnosis=y_test.values
    ).to_csv(OUTPUT_TEST, index=False)

if __name__ == "__main__":
    main()
