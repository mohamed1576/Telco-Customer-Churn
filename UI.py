import streamlit as st
import pandas as pd
import joblib
import numpy as np
import sys
sys.modules['np'] = np
import plotly.express as px
from sklearn.base import BaseEstimator , TransformerMixin


class TotalChargesTransformer(BaseEstimator , TransformerMixin):
    def __init__(self):
        pass

    def fit(self , x , y=None):
        return self
    
    def transform(self , x):
        x_copy = x.copy()

        if "TotalCharges" in x_copy.columns:
            x_copy["TotalCharges"] = x_copy["TotalCharges"].replace(r'^\s*$' ,  np.nan , regex=True)
            x_copy["TotalCharges"] = x_copy["TotalCharges"].astype(float)
            x_copy["TotalCharges"] = x_copy["TotalCharges"].fillna(0.0)

        return x_copy
    

class ServiceAggregator (BaseEstimator , TransformerMixin):
    def __init__(self , services_columns):
        self.services_columns = services_columns

    def fit(self , x , y = None):
        return self
    
    def transform (self , x):
        x_copy = x.copy()


        x_copy["total_serveces"] = x_copy[self.services_columns].apply(

            lambda row : sum((row != "No") & (row != "No phone service" )), axis= 1
        )

        if 'MonthlyCharges' in x_copy.columns :
            x_copy["cost_per_secvice"] = x_copy["MonthlyCharges"] / (x_copy["TotalCharges"] + 0.1)

        return x_copy    
    
st.set_page_config(page_title="Telecom Churn Predictor" , layout="wide")

@st.cache_resource
def load_my_model () :
   return joblib.load("optimized_telecom_churn_model.pkl")

model = load_my_model()    

st.sidebar.title("Control Panel")

selected_app_mode= st.sidebar.selectbox("Choose Prediction Mode" , ["Single Customer Entry", "Batch Processing (Upload CSV)"])

if selected_app_mode == "Single Customer Entry":

    tab1 , tab2 , tab3 = st.tabs([" Financials & Contract" , " Services Subscribed" , " Customer Profile"])

    with tab1 :
        st.header("Contract & Billing Details")
        contract = st.selectbox("Contract Type" , ["Month-to-month" , "One year" , "Two year"])
        tenure = st.slider ("Tenure (Months with company)" , min_value=0 , max_value = 72 , value = 12)
        monthly_charges = st.number_input("Monthly Charges ($)" , min_value=0.0 , value=50.0)
        total_charges = st.number_input("Total Charges ($)" , min_value=0.0 , value=600.0)

    with tab2 : 
        st.header("Core & Internet Services")
        internet_service = st.selectbox("Internet Service Provider", ['DSL', 'Fiber optic', 'No'])
        phone_service = st.selectbox("Phone Service", ['Yes', 'No'])
        multiple_lines = st.selectbox("Multiple Lines", ['Yes', 'No', 'No phone service'])

        st.subheader("Add-on Features & Security")
        online_security = st.selectbox ("Online Security", ['Yes', 'No', 'No internet service'])
        online_backup = st.selectbox ("Online Backup", ['Yes', 'No', 'No internet service'])
        device_protection = st.selectbox ("Device Protection", ['Yes', 'No', 'No internet service'])
        tech_support = st.selectbox ("Tech Support", ['Yes', 'No', 'No internet service'])
        streaming_tv = st.selectbox  ("Streaming TV", ['Yes', 'No', 'No internet service'])
        streaming_movies = st.selectbox ("Streaming Movies", ['Yes', 'No', 'No internet service'])

    with tab3 :
        st.header("Demographics & Payment Preferences")
        gender = st.selectbox  ("Gender", ['Male', 'Female'])
        senior_citizen_raw = st.selectbox ("Is the customer a Senior Citizen?", ['No', 'Yes'])

        senior_citizen_raw = 1 if senior_citizen_raw == "Yes" else 0

        partner = st.selectbox("Has a Partner?", ['Yes', 'No'])
        dependents = st.selectbox("Has Dependents?", ['Yes', 'No'])
        paperless_billing = st.selectbox ("Paperless Billing", ['Yes', 'No'])

        payment_method = st.selectbox ("Payment Method", [
            'Electronic check', 
            'Mailed check', 
            'Bank transfer (automatic)', 
            'Credit card (automatic)'
            ])


    input_data = {
        'customerID': '0000-MOCK', 
        'gender': gender,
        'SeniorCitizen': senior_citizen_raw,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    input_df = pd.DataFrame([input_data])

    st.markdown("---")


    if st.button("Analyze Retention Risk", use_container_width=True):
    
        probability = model.predict_proba(input_df)[0][1] * 100

   
        if probability > 50:
            st.error(f" High Churn Risk Detected!** The probability of this customer leaving is {probability:.2f}%")
        else:
            st.success(f" Customer is Loyal & Stable.** The churn probability is very low at {probability:.2f}%")

elif selected_app_mode == "Batch Processing (Upload CSV)":
    st.title("Customer Churn Analysis")
    st.write("Upload a corporate CSV sheet to extract big-picture churn statistics.")
    
    
    uploaded_csv_file = st.file_uploader("Upload Telecom Data Sheet (CSV Only)" , type=["csv"])

    if uploaded_csv_file is not None : 
        df = pd.read_csv(uploaded_csv_file)
        if 'Churn' in df.columns:
            df_features = df.drop(columns=['Churn'])
        else:
            df_features = df.copy()

        with st.spinner("Executing pipeline predictions..."):
            probs = model.predict_proba(df_features)[:, 1] * 100
            preds = (probs > 50).astype(int)
         
            df['Churn Probability (%)'] = probs
            df["Risk Evaluation"] = preds
            df["Risk Evaluation"] = df['Risk Evaluation'].map(
              {1 : "Risk Evaluation" , 0: 'Loyal & Stable'}
            )
        st.success("Dataset successfully analyzed!")

        total_users = len(df)
        churn_users = (preds == 1).sum()
        churn_rate = (churn_users / total_users)*100

        col1 , col2 , col3 = st.columns(3)
        col1.metric("Total Customers Evaluated" ,total_users )
        col2.metric("High Risk Customers Found" , churn_users , delta=f"churn_rate: {churn_rate} " , delta_color="inverse")
        col3.metric("Average Churn Probability" , f"{probs.mean():.2f}%") 

        st.subheader("Visual Breakdown of Retention Risks")
        fig = px.pie (
            df,
            names="Risk Evaluation",
            color="Risk Evaluation",
            color_discrete_map={"High Churn Risk" : '#ff4b4b' , 'loyal' : "#009688"},
            hole=0.4
        )

        st.plotly_chart(fig, use_container_width=True)

        st.subheader(" Sorted Strategic Risk Sheet")
        sorted_df = df.sort_values(by='Churn Probability (%)', ascending=False)
        st.dataframe(sorted_df)

        csv_out = sorted_df.to_csv(index=False).encode('utf-8')
        st.download_button(
        label=" Export Annotated Predictions CSV Sheet",
        data=csv_out,
        file_name="telecom_churn_bulk_predictions.csv",
        mime="text/csv",
        use_container_width=True
        )

