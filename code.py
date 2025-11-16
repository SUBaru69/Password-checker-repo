"""
PASSWORD STRENGTH ANALYZER - FULL PROJECT (ONE FILE)
----------------------------------------------------
Includes:

1. Password generation
2. Entropy calculation
3. Feature extraction
4. Password labeling rules
5. Dataset generation
6. Model training + evaluation
7. Model saving
8. Final interactive analyzer WITH FEEDBACK SYSTEM

Modules needed:
    pip install numpy pandas scikit-learn joblib
"""

import random
import string
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os



# 1. PASSWORD GENERATION (Synthetic only)


def random_password():
    chars = string.ascii_letters + string.digits + string.punctuation
    length = random.randint(4, 25)
    return ''.join(random.choice(chars) for _ in range(length))



# 2. ENTROPY CALCULATION


def compute_entropy(pwd):
    unique = len(set(pwd))
    length = len(pwd)
    if unique <= 1:
        return 0.0
    return float(length * np.log2(unique))



# 3. FEATURE EXTRACTION


def extract_features(pwd):
    length = len(pwd)
    lowers = sum(1 for c in pwd if c.islower())
    uppers = sum(1 for c in pwd if c.isupper())
    digits = sum(1 for c in pwd if c.isdigit())
    specials = sum(1 for c in pwd if c in string.punctuation)
    variety = sum([lowers > 0, uppers > 0, digits > 0, specials > 0])
    entropy = compute_entropy(pwd)
    return [length, lowers, uppers, digits, specials, variety, entropy]



# 4. LABELING RULES


def label_password(pwd):
    length = len(pwd)
    lowers = any(c.islower() for c in pwd)
    uppers = any(c.isupper() for c in pwd)
    digits = any(c.isdigit() for c in pwd)
    specials = any(c in string.punctuation for c in pwd)
    variety = sum([lowers, uppers, digits, specials])
    entropy = compute_entropy(pwd)

    # WEAK
    if length < 8 or variety <= 1 or entropy < 20:
        return 0

    # STRONG
    if length >= 10 and variety >= 3 and entropy >= 30:
        return 2

    # MEDIUM
    return 1



# 5. FEEDBACK SYSTEM 


def password_feedback(pwd):
    """Provides detailed feedback on what the password is missing."""
    feedback = []

    has_lower = any(c.islower() for c in pwd)
    has_upper = any(c.isupper() for c in pwd)
    has_digit = any(c.isdigit() for c in pwd)
    has_special = any(c in string.punctuation for c in pwd)
    length = len(pwd)
    entropy_value = compute_entropy(pwd)

    # CATEGORY CHECKS 
    if not has_lower:
        feedback.append("Missing lowercase letters.")
    if not has_upper:
        feedback.append("Missing uppercase letters.")
    if not has_digit:
        feedback.append("Missing digits (0–9).")
    if not has_special:
        feedback.append("Missing special characters (!@#$ etc.).")

    # LENGTH CHECK 
    if length < 8:
        feedback.append("Password is too short (minimum 8 characters).")

    # ENTROPY CHECK 
    if entropy_value < 20:
        feedback.append("Entropy too low (password is too predictable).")

    #  VARIETY CHECK 
    variety = sum([has_lower, has_upper, has_digit, has_special])
    if variety <= 2:
        feedback.append("Not enough character variety (add more different types).")

    # If nothing missing 
    if not feedback:
        feedback.append("Password meets all strength requirements.")

    return feedback



# 6. DATASET GENERATION


def generate_dataset(size=3000, file_name="password_dataset.csv"):
    passwords = []
    strengths = []

    for _ in range(size):
        pwd = random_password()
        passwords.append(pwd)
        strengths.append(label_password(pwd))

    df = pd.DataFrame({"password": passwords, "strength": strengths})
    df.to_csv(file_name, index=False)

    print(f"Dataset saved as {file_name} with {len(df)} rows.")
    return df



# 7. TRAIN MODEL


def train_model(dataset_path="password_dataset.csv",
                model_path="password_strength_model.pkl"):

    if not os.path.exists(dataset_path):
        df = generate_dataset()
    else:
        df = pd.read_csv(dataset_path)

    X = [extract_features(p) for p in df["password"]]
    y = df["strength"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("\nModel Evaluation:")
    print("Accuracy:", accuracy_score(y_test, preds))
    print("\nClassification Report:")
    print(classification_report(y_test, preds))

    joblib.dump(model, model_path)
    print(f"\nModel saved as: {model_path}")

    return model


# 8. PREDICTION / ANALYZER

def predict_strength(model, pwd):
    features = extract_features(pwd)
    pred = model.predict([features])[0]
    probs = model.predict_proba([features])[0]
    labels = ["Weak", "Medium", "Strong"]
    return labels[int(pred)], probs


def interactive_analyzer(model):
    print("\nPassword Strength Analyzer Ready!")
    print("Type a password (or 'exit' to quit)\n")

    while True:
        pwd = input("Enter password: ")

        if pwd.lower().strip() == "exit":
            break

        label, probs = predict_strength(model, pwd)

        print(f"\nPrediction : {label}")
        print("Confidence :", probs.round(3).tolist())

        print("\nFeedback:")
        for msg in password_feedback(pwd):
            print(" -", msg)

        print("--------------------------------------\n")



# 9. MAIN


if __name__ == "__main__":
    print("\n=== TRAINING MODEL ===")
    model = train_model()

    print("\n=== STARTING ANALYZER ===")
    interactive_analyzer(model)
