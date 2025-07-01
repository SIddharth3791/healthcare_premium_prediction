import pandas as pd
from joblib import load

model_large_group = load("artifacts/premium_prediction_model_larger_group.joblib")
model_young_group = load("artifacts/premium_prediction_model_young_group.joblib")

scaler_large_group = load("artifacts/scaler_premium_prediction_model_larger_group.joblib")
scaler_young_group = load("artifacts/scaler_premium_prediction_young_group.joblib")


def predict(user_input_dict):
    user_input_df = pre_process_user_input(user_input_dict)
    if user_input_dict['Age'] <= 25:
        prediction = model_young_group.predict(user_input_df)
    else:
        prediction = model_large_group.predict(user_input_df)
    return int(prediction)

def pre_process_user_input(user_input_dict):
    # Consider moving to artifiact as a file and calling
    expected_columns = [
        'age', 'number_of_dependants', 'income_lakhs', 'insurance_plan', 'genetical_risk', 'normalized_risk_score',
        'gender_Male', 'region_Northwest', 'region_Southeast', 'region_Southwest', 'marital_status_Unmarried',
        'bmi_category_Obesity', 'bmi_category_Overweight', 'bmi_category_Underweight', 'smoking_status_Occasional',
        'smoking_status_Regular', 'employment_status_Salaried', 'employment_status_Self-Employed'
    ]

    insurance_plan_encoding = {'Bronze': 1, 'Silver': 2, 'Gold': 3}
    df = pd.DataFrame(0, columns=expected_columns, index=[0])

    category_mappings = {
        'Gender': {'Male': 'gender_Male'},
        'Region': {
            'Northwest': 'region_Northwest',
            'Southeast': 'region_Southeast',
            'Southwest': 'region_Southwest'
        },
        'Marital Status': {'Unmarried': 'marital_status_Unmarried'},
        'BMI Category': {
            'Obesity': 'bmi_category_Obesity',
            'Overweight': 'bmi_category_Overweight',
            'Underweight': 'bmi_category_Underweight'
        },
        'Smoking Status': {
            'Occasional': 'smoking_status_Occasional',
            'Regular': 'smoking_status_Regular'
        },
        'Employment Status': {
            'Salaried': 'employment_status_Salaried',
            'Self-Employed': 'employment_status_Self-Employed'
        }
    }

    # Iterate through the user input dictionary
    for key, value in user_input_dict.items():
        # Handle categorical mappings
        if key in category_mappings and value in category_mappings[key]:
            df[category_mappings[key][value]] = 1
        # Handle specific keys with direct assignments
        elif key == 'Insurance Plan':
            df['insurance_plan'] = insurance_plan_encoding.get(value, 1)
        elif key == 'Age':
            df['age'] = value
        elif key == 'Number of Dependants':
            df['number_of_dependants'] = value
        elif key == 'Income in Lakhs':
            df['income_lakhs'] = value
        elif key == 'Genetical Risk':
            df['genetical_risk'] = value

    # Assuming the 'normalized_risk_score' needs to be calculated based on the 'age'
    df['normalized_risk_score'] = calculate_normalized_risk(user_input_dict['Medical History'])
    scaled_data_frame = handle_scaling(user_input_dict['Age'], df)

    return scaled_data_frame

def calculate_normalized_risk(medical_history):
    risk_scores = {
        "diabetes": 6,
        "heart disease": 8,
        "high blood pressure": 6,
        "thyroid": 5,
        "no disease": 0,
        "none": 0
    }
    # Split the medical history into potential two parts and convert to lowercase
    diseases = medical_history.lower().split(" & ")

    # Calculate the total risk score by summing the risk scores for each part
    total_risk_score = sum(risk_scores.get(disease, 0) for disease in diseases)  # Default to 0 if disease not found

    max_score = 14 # risk score for heart disease (8) + second max risk score (6) for diabetes or high blood pressure
    min_score = 0  # Since the minimum score is always 0

    # Normalize the total risk score
    normalized_risk_score = (total_risk_score - min_score) / (max_score - min_score)

    return normalized_risk_score

def handle_scaling(age, df):
    if age <= 25:
        scaler_object = scaler_young_group
    else:
        scaler_object = scaler_large_group

    cols_to_scale = scaler_object['cols_to_scale']
    scaler = scaler_object['scaler']

    df['income_level'] = None # since scaler object expects income_level supply it. This will have no impact on anything
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    df.drop('income_level', axis='columns', inplace=True)

    return df

