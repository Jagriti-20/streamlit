import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

# Set page configuration
st.set_page_config(page_title="Health Prediction System", layout="wide")

# Perceptron function for CKD prediction
def perceptron(data, theta, n, real_y):
    b = 0
    for i in range(n):
        temp = np.dot(theta, data[i]) + b
        if (temp >= 0 and real_y[i] == 0):
            theta -= data[i]
            b -= 1
        elif (temp < 0 and real_y[i] == 1):
            theta += data[i]
            b += 1
    return theta, b

# Load dataset and preprocess for CKD prediction
dataset = pd.read_csv("data/datasetperceptron.csv")
x = dataset.iloc[:, [10, 11, 12, 20]].values
y = dataset.iloc[:, 25].values
n = len(y)

for i in range(n):
    x[i][3] = 1 if x[i][3] == 'yes' else 0
    y[i] = 1 if y[i] == 'ckd' else 0

data = x.astype(np.float64)
real_y = y.astype(np.int32)
means = data.mean(axis=0)
stds = [560, 406, 49, 1]
data = (data - means) / stds
theta = np.array([0.1, 0.2, 0.2, 0.1])
theta, b = perceptron(data, theta, n, real_y)

# Streamlit app
st.title("Health Prediction System")

# Function to plot a bar chart
def plot_input_values(values, labels):
    plt.figure(figsize=(8, 4))
    plt.bar(labels, values, color='skyblue')
    plt.xlabel('Input Features')
    plt.ylabel('Values')
    plt.title('Input Values for Prediction')
    plt.xticks(rotation=45)
    st.pyplot(plt)

# CKD Prediction Page
def ckd_prediction_page():
    st.image("https://cdn-icons-png.freepik.com/256/7273/7273415.png?uid=R179575664&ga=GA1.1.1804309978.1734442655&semt=ais_hybrid", width=100)  # Replace with your CKD icon URL
    st.header("🩺Chronic Kidney Disease Prediction")
    st.write("Enter the following details to predict if you have chronic kidney disease.")

    # User input for CKD prediction
    age = st.number_input("Age:", min_value=1, step=1)
    gender = st.selectbox("Gender:", ["male", "female"])
    complexion = st.selectbox("Complexion:", ["black", "other"])
    v1 = st.number_input("Blood Glucose Random:", step=0.1)
    v2 = st.number_input("Blood Urea:", step=0.1)
    v3 = st.number_input("Serum Creatinine:", step=0.1)
    v4 = st.selectbox("Diabetic Mellitus (1: Yes, 0: No):", [1, 0])

    if st.button("Predict CKD"):
        # Normalize inputs
        v11 = (v1 - means[0]) / stds[0]
        v12 = (v2 - means[1]) / stds[1]
        v13 = (v3 - means[2]) / stds[2]
        v14 = (v4 - means[3]) / stds[3]
        
        # Perceptron prediction
        x_input = np.dot(theta, [v11, v12, v13, v14]) + b
        
        # eGFR calculation
        if gender == 'male' and complexion == 'other':
            eGFR = 186 * (v3 ** -1.154) * (age ** -0.203)
        elif gender == 'female' and complexion == 'black':
            eGFR = 186 * (v3 ** -1.154) * (age ** -0.203) * 0.742 * 1.210
        elif gender == 'female' and complexion == 'other':
            eGFR = 186 * (v3 ** -1.154) * (age ** -0.203) * 0.742
        else:
            eGFR = 186 * (v3 ** -1.154) * (age ** -0.203) * 1.210
        
        # Output based on perceptron prediction
        if x_input < 0:
            prediction = "You are not affected by chronic kidney disease. Your kidney is fine."
            eGFR_info = ""
        else:
            prediction = "You are affected by chronic kidney disease."
            
            # Kidney description based on eGFR value
            if eGFR > 90:
                eGFR_info = "Normal kidney function but urine findings or structural abnormalities or genetic trait point to kidney disease."
            elif eGFR > 60:
                eGFR_info = "Mildly reduced kidney function."
            elif eGFR > 30:
                eGFR_info = "Moderately reduced kidney function."
            elif eGFR > 15:
                eGFR_info = "Severely reduced kidney function."
            else:
                eGFR_info = "Very severe, or end-stage kidney failure."
            
            eGFR_info += f"\nYour eGFR is {eGFR:.2f}"

        # Display results
        st.write(prediction)
        if eGFR_info:
            st.write(eGFR_info)

        # Plot input values
        plot_input_values([age, v1, v2, v3, v4], ['Age', 'Blood Glucose', 'Blood Urea', 'Serum Creatinine', 'Diabetic Mellitus'])

def diabetes_risk_page():
    st.image("https://cdn-icons-png.freepik.com/256/11210/11210018.png?uid=R179575664&ga=GA1.1.1804309978.1734442655&semt=ais_hybrid", width=100)  # Replace with your Diabetes icon URL
    st.header("🍭Diabetes Risk Prediction")
    st.write("Enter the following details to predict your diabetes risk.")

    # User input for diabetes risk prediction
    age = st.number_input("Age:", min_value=1, step=1)
    bmi = st.number_input("Body Mass Index (BMI):", step=0.1)
    glucose = st.number_input("Glucose Level:", step=0.1)
    insulin = st.number_input("Insulin Level:", step=0.1)

    if st.button("Predict Diabetes Risk"):
        # Adjusted risk calculation
        risk_score = (age * 0.02) + (bmi * 0.1) + (glucose * 0.05) + (insulin * 0.1)
        
        # Display the calculated risk score for debugging
        st.write(f"Calculated Risk Score: {risk_score:.2f}")

        if risk_score > 15:
            prediction = "You are at high risk of diabetes."
        elif risk_score > 10:
            prediction = "You are at moderate risk of diabetes."
        else:
            prediction = "You are at low risk of diabetes."

        st.write(prediction)

        # Plot input values
        plot_input_values([age, bmi, glucose, insulin], ['Age', 'BMI', 'Glucose', 'Insulin'])

# Hypertension Risk Prediction Page
def hypertension_risk_page():
    st.image("https://cdn-icons-png.freepik.com/256/18519/18519832.png?uid=R179575664&ga=GA1.1.1804309978.1734442655&semt=ais_hybrid", width=100)  # Replace with your Hypertension icon URL
    st.header("👩‍⚕Hypertension Risk Prediction")
    st.write("Enter the following details to predict your hypertension risk.")

    # User input for hypertension risk prediction
    age = st.number_input("Age:", min_value=1, step=1)
    systolic = st.number_input("Systolic Blood Pressure:", step=1)
    diastolic = st.number_input("Diastolic Blood Pressure:", step=1)
    weight = st.number_input("Weight (kg):", step=0.1)

    if st.button("Predict Hypertension Risk"):
        # Simple risk calculation (example logic)
        risk_score = (systolic - 120) + (diastolic - 80) + (age * 0.5) + (weight * 0.2)
        
        if risk_score > 20:
            prediction = "You are at high risk of hypertension."
        elif risk_score > 10:
            prediction = "You are at moderate risk of hypertension."
        else:
            prediction = "You are at low risk of hypertension."

        st.write(prediction)

        # Plot input values
        plot_input_values([age, systolic, diastolic, weight], ['Age', 'Systolic BP', 'Diastolic BP', 'Weight'])

# Cardiovascular Risk Prediction ```python
def cardiovascular_risk_page():
    st.image("https://cdn-icons-png.freepik.com/256/1047/1047843.png?uid=R179575664&ga=GA1.1.1804309978.1734442655&semt=ais_hybrid", width=100)  # Replace with your Cardiovascular icon URL
    st.header("💓Cardiovascular Risk Prediction")
    st.write("Enter the following details to predict your cardiovascular risk.")

    # User input for cardiovascular risk prediction
    age = st.number_input("Age:", min_value=1, step=1)
    cholesterol = st.number_input("Cholesterol Level:", step=1)
    blood_pressure = st.number_input("Blood Pressure:", step=1)
    smoking = st.selectbox("Do you smoke? (1: Yes, 0: No)", [1, 0])

    if st.button("Predict Cardiovascular Risk"):
        # Simple risk calculation (example logic)
        risk_score = (cholesterol / 200) + (blood_pressure / 120) + (age * 0.1) + (smoking * 5)
        
        if risk_score > 10:
            prediction = "You are at high risk of cardiovascular disease."
        elif risk_score > 5:
            prediction = "You are at moderate risk of cardiovascular disease."
        else:
            prediction = "You are at low risk of cardiovascular disease."

        st.write(prediction)

        # Plot input values
        plot_input_values([age, cholesterol, blood_pressure, smoking], ['Age', 'Cholesterol', 'Blood Pressure', 'Smoking'])

# Main app logic
def main():
    menu = ["CKD Prediction", "Diabetes Risk", "Hypertension Risk", "Cardiovascular Risk"]
    choice = st.sidebar.selectbox("Select a prediction model", menu)

    if choice == "CKD Prediction":
        ckd_prediction_page()
    elif choice == "Diabetes Risk":
        diabetes_risk_page()
    elif choice == "Hypertension Risk":
        hypertension_risk_page()
    elif choice == "Cardiovascular Risk":
        cardiovascular_risk_page()

if __name__ == "__main__":  # Corrected line
    main()
