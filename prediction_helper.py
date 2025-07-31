import joblib
import numpy as np
import pandas as pd

MODEL_PATH = 'artifacts/model_data.joblib'

model_data = joblib.load(MODEL_PATH)
model = model_data['model']
scaler = model_data['scaler']
features = model_data['features']
cols_to_scale = model_data['cols_to_scale']


def encode_category(value, categories):
    return {f"{categories}_{cat}": int(value == cat) for cat in ['Owned', 'Rented'] if cat in categories} if categories == 'residence_type' else \
           {f"{categories}_{cat}": int(value == cat) for cat in ['Education', 'Home', 'Personal'] if cat in categories} if categories == 'loan_purpose' else \
           {f"{categories}_Unsecured": int(value == 'Unsecured')}


def prepare_input(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
                  delinquency_ratio, credit_utilization_ratio, num_open_accounts,
                  residence_type, loan_purpose, loan_type):

    base_input = {
        'age': age,
        'loan_tenure_months': loan_tenure_months,
        'number_of_open_accounts': num_open_accounts,
        'credit_utilization_ratio': credit_utilization_ratio,
        'loan_to_income': loan_amount / income if income > 0 else 0,
        'delinquency_ratio': delinquency_ratio,
        'avg_dpd_per_delinquency': avg_dpd_per_delinquency
    }

    category_encoded = {}
    category_encoded.update(encode_category(residence_type, 'residence_type'))
    category_encoded.update(encode_category(loan_purpose, 'loan_purpose'))
    category_encoded.update(encode_category(loan_type, 'loan_type'))

    dummy_fillers = {col: 1 for col in [
        'number_of_dependants', 'years_at_current_address', 'zipcode',
        'sanction_amount', 'processing_fee', 'gst', 'net_disbursement',
        'principal_outstanding', 'bank_balance_at_application',
        'number_of_closed_accounts', 'enquiry_count'
    ]}

    full_input = {**base_input, **category_encoded, **dummy_fillers}
    df = pd.DataFrame([full_input])
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
    df = df[features]

    return df


def calculate_credit_score(input_df, base_score=300, scale_length=600):
    x = np.dot(input_df.values, model.coef_.T) + model.intercept_
    default_probability = 1 / (1 + np.exp(-x))
    non_default_probability = 1 - default_probability
    credit_score = base_score + non_default_probability.flatten() * scale_length

    score = credit_score[0]

    if 300 <= score < 500:
        rating = 'Poor'
    elif 500 <= score < 650:
        rating = 'Average'
    elif 650 <= score < 750:
        rating = 'Good'
    elif 750 <= score <= 900:
        rating = 'Excellent'
    else:
        rating = 'Undefined'

    return default_probability.flatten()[0], int(score), rating


def predict(age, income, loan_amount, loan_tenure_months, avg_dpd_per_delinquency,
            delinquency_ratio, credit_utilization_ratio, num_open_accounts,
            residence_type, loan_purpose, loan_type):

    input_df = prepare_input(age, income, loan_amount, loan_tenure_months,
                             avg_dpd_per_delinquency, delinquency_ratio,
                             credit_utilization_ratio, num_open_accounts,
                             residence_type, loan_purpose, loan_type)

    return calculate_credit_score(input_df)
