import os
import pickle
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)


def generate_synthetic_data(num_patients: int = 30000) -> pd.DataFrame:
    """
    Generates synthetic patient health data with realistic distributions and balanced risk categories.

    :param num_patients:  Number of synthetic patient records to generate.
    :return: A DataFrame containing the generated patient health data with risk categories.
    """
    # Generate age using a normal distribution centered around 55 (typical menopause age range). The scale sets the
    # standard deviation, controlling how spread out the values are. Most values are between 50-60 at 5 standard
    # deviations, but clip any values outside of the 45-60 range.
    age = np.random.normal(loc=55, scale=5, size=num_patients).astype(int)
    age = np.clip(age, 45, 65)  # Ensure age is within realistic bounds

    # BMI follows a slightly skewed normal distribution, more cases in the overweight range
    bmi = np.random.normal(loc=28, scale=4, size=num_patients)
    bmi = np.clip(bmi, 20, 40)  # Clip to valid BMI range

    # Blood Pressure: Increases with BMI and age
    bp_systolic = np.random.normal(loc=130 + (bmi - 25) * 0.8, scale=10, size=num_patients)
    bp_systolic = np.clip(bp_systolic, 110, 160)

    # Cholesterol: Increases with BMI and age. BMI above 25 (overweight range) is associated with higher cholesterol
    # levels. The 1.5 multiplier means that for every 1 unit increase in BMI beyond 25, Cholesterol increases by 1.5 mg/dL.
    # Aging tends to increase cholesterol, but not as drastically as BMI. The 0.5 multiplier means that for every year
    # beyond 50, cholesterol increases by 0.5 mg/dL.
    cholesterol_ldl = np.random.normal(loc=150 + (bmi - 25) * 1.5 + (age - 50) * 0.5, scale=15, size=num_patients)
    cholesterol_ldl = np.clip(cholesterol_ldl, 100, 190)

    # Bone Density: Lower for older individuals
    bone_density_t_score = np.random.normal(loc=-1.5 - (age - 50) * 0.1, scale=0.7, size=num_patients)
    bone_density_t_score = np.clip(bone_density_t_score, -3.0, 0.5)

    # Heart Rate Variability: Lower with high BP and age
    heart_rate_variability = np.random.normal(
        loc=40 - (bp_systolic - 120) * 0.2 - (age - 50) * 0.2, scale=5, size=num_patients)
    heart_rate_variability = np.clip(heart_rate_variability, 20, 50)

    # Steps Per Day: Decreases with age and BMI
    steps_per_day = np.random.normal(
        loc=8000 - (bmi - 25) * 300 - (age - 50) * 50, scale=2000, size=num_patients)
    steps_per_day = np.clip(steps_per_day, 2000, 12000)

    # Create DataFrame
    df = pd.DataFrame({
        "Age": age,
        "BMI": np.round(bmi, 1),
        "Blood_Pressure_Systolic": bp_systolic.astype(int),
        "Cholesterol_LDL": cholesterol_ldl.astype(int),
        "Bone_Density_T_Score": np.round(bone_density_t_score, 2),
        "Heart_Rate_Variability": heart_rate_variability.astype(int),
        "Steps_Per_Day": steps_per_day.astype(int)
    })

    # Compute risk scores based on a weighted combination of features
    df["Cardio_Risk_Score"] = (
            (bp_systolic > 140).astype(int) * 2 +
            (cholesterol_ldl > 160).astype(int) * 2 +
            (bmi > 30).astype(int) * 1 +
            (heart_rate_variability < 30).astype(int) * 1
    )
    df["Osteoporosis_Risk_Score"] = (
            (bone_density_t_score < -2.5).astype(int) * 2 +
            (age > 60).astype(int) * 1 +
            (steps_per_day < 5000).astype(int) * 1
    )

    # Assign risk categories by defining threshold bins by converting the continuous Cardio_Risk_Score into discrete
    # risk categories (Low=1, Moderate=2, High=3). The bins=[1, 3] define two cutoffs:
    #  - Below 1 → Category 0 (Low Risk)
    #  - Between 1 and 3 → Category 1 (Moderate Risk)
    #  - Above 3 → Category 2 (High Risk)
    df["Cardio_Risk"] = np.digitize(df["Cardio_Risk_Score"], bins=[1, 3], right=True)
    df["Osteoporosis_Risk"] = np.digitize(df["Osteoporosis_Risk_Score"], bins=[1, 3], right=True)

    # Drop risk score columns after assigning categories
    df.drop(["Cardio_Risk_Score", "Osteoporosis_Risk_Score"], axis=1, inplace=True)

    return df


def train_models() -> None:
    """
    Trains two Random Forest models for cardiovascular and osteoporosis risk classification.
    Saves trained models and test datasets.
    """
    df = generate_synthetic_data(30000)

    features = [
        "Age",
        "BMI",
        "Blood_Pressure_Systolic",
        "Cholesterol_LDL",
        "Bone_Density_T_Score",
        "Heart_Rate_Variability",
        "Steps_Per_Day"
    ]
    # X is the feature matrix (a structured dataset where each row is a sample, and each
    # column is a feature). It excludes the target labels (Cardio_Risk and Osteoporosis_Risk),
    # meaning it only contains independent variables.
    X = df[features]

    # These lines extract the target labels (also called dependent variables) for model
    # training. These two are the values we want the model to predict based on X (the
    # input features). The machine learning model takes X (input features) and learns
    # to predict y (target labels)
    y_cardio = df["Cardio_Risk"]
    y_osteo = df["Osteoporosis_Risk"]

    # Randomly splits the dataset into:
    #  - Training Data (X_train_cardio, y_train_cardio) → Used to train the model.
    #  - Test Data (X_test_cardio, y_test_cardio) → Used to evaluate the model's performance
    X_train_cardio, X_test_cardio, y_train_cardio, y_test_cardio = train_test_split(
        X, y_cardio, test_size=0.2, random_state=42
    )
    X_train_osteo, X_test_osteo, y_train_osteo, y_test_osteo = train_test_split(
        X, y_osteo, test_size=0.2, random_state=42
    )

    # Train Random Forest Classifiers with 100 decision trees (n_estimators=100). The final prediction is made by
    # aggregating the predictions of all 100 trees (majority vote for classification).
    cardio_model = RandomForestClassifier(n_estimators=100, random_state=42)
    cardio_model.fit(X_train_cardio, y_train_cardio)

    osteo_model = RandomForestClassifier(n_estimators=100, random_state=42)
    osteo_model.fit(X_train_osteo, y_train_osteo)

    # Save trained models
    with open("cardio_model.pkl", "wb") as f:
        pickle.dump(cardio_model, f)
    with open("osteo_model.pkl", "wb") as f:
        pickle.dump(osteo_model, f)

    # Save test datasets for evaluation
    with open("test_data.pkl", "wb") as f:
        pickle.dump({"X_test": X_test_cardio, "y_test_cardio": y_test_cardio, "y_test_osteo": y_test_osteo}, f)

    print("Training complete. Models saved.")


def load_model(file_path):
    """
    Load the model from disk
    :param file_path: The path where the model resides
    :return: The model
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def evaluate_models():
    """
    Loads the trained models and evaluates their performance on the test dataset. Outputs accuracy and a classification
    report for both the cardiovascular and osteoporosis models.
    """
    # Load test data
    with open("test_data.pkl", "rb") as f:
        test_data = pickle.load(f)

    X_test = test_data["X_test"]
    y_test_cardio = test_data["y_test_cardio"]
    y_test_osteo = test_data["y_test_osteo"]

    # Load trained models
    with open("cardio_model.pkl", "rb") as f:
        cardio_model = pickle.load(f)
    with open("osteo_model.pkl", "rb") as f:
        osteo_model = pickle.load(f)

    # Make predictions
    y_pred_cardio = cardio_model.predict(X_test)
    y_pred_osteo = osteo_model.predict(X_test)

    # Print evaluation metrics
    print("=== Cardiovascular Risk Model ===")
    print("Accuracy:", accuracy_score(y_test_cardio, y_pred_cardio))
    print("Classification Report:\n", classification_report(y_test_cardio, y_pred_cardio))

    print("\n=== Osteoporosis Risk Model ===")
    print("Accuracy:", accuracy_score(y_test_osteo, y_pred_osteo))
    print("Classification Report:\n", classification_report(y_test_osteo, y_pred_osteo))

app = Flask(__name__)


@app.route('/predict_risk', methods=['POST'])
def predict_risk():
    data = request.json
    input_data = np.array([[
        data["Age"],
        data["BMI"],
        data["Blood_Pressure_Systolic"],
        data["Cholesterol_LDL"],
        data["Bone_Density_T_Score"],
        data["Heart_Rate_Variability"],
        data["Steps_Per_Day"]
    ]])

    cardio_model = load_model("cardio_model.pkl")
    osteo_model = load_model("osteo_model.pkl")

    cardio_risk = cardio_model.predict(input_data)[0]
    osteo_risk = osteo_model.predict(input_data)[0]

    risk_mapping = {0: "Low", 1: "Moderate", 2: "High"}
    response = {
        "Predicted_Cardio_Risk": risk_mapping[cardio_risk],
        "Predicted_Osteo_Risk": risk_mapping[osteo_risk],
        "Recommendations": []
    }

    if cardio_risk == 2:
        response["Recommendations"].append(
            "⚠️ High Cardiovascular Risk: Consider a heart-healthy diet, increase physical activity, and consult a cardiologist.")
    elif cardio_risk == 1:
        response["Recommendations"].append(
            "🟡 Moderate Cardiovascular Risk: Monitor blood pressure and cholesterol levels, engage in regular exercise, and follow a balanced diet.")

    if osteo_risk == 2:
        response["Recommendations"].append(
            "⚠️ High Osteoporosis Risk: Increase calcium and vitamin D intake, start weight-bearing exercises, and consult an endocrinologist for bone health management.")
    elif osteo_risk == 1:
        response["Recommendations"].append(
            "🟡 Moderate Osteoporosis Risk: Monitor bone density, ensure adequate calcium intake, and engage in regular strength training.")

    return jsonify(response)


@app.route('/retrain_models', methods=['POST'])
def retrain_models():
    train_models()
    return jsonify({"message": "Models retrained successfully!"})


if __name__ == '__main__':
    if (
        not os.path.exists("cardio_model.pkl") or
        not os.path.exists("osteo_model.pkl") or
        not os.path.exists("test_data.pkl")
    ):
        train_models()
    else:
        # Run the evaluation
        evaluate_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
