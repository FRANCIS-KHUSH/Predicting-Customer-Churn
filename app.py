import streamlit as st
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import os

# Load the trained model
filename = 'Models/churn_prediction_model_selected_features.sav'
loaded_model = pickle.load(open(filename, 'rb'))

# Streamlit app
st.title('Customer Churn Prediction App')

st.write("""
### Predict if a customer will churn based on key engagement metrics.
""")

# Sidebar with additional options
st.sidebar.title("Churn Prediction Options")
model_choice = st.sidebar.selectbox("Choose a prediction model:", 
                                    ("Logistic Regression", "Random Forest"))

st.sidebar.write("Adjust the sliders to change the input metrics.")

# Create sliders and interactive inputs for user data
nps = st.sidebar.slider('Net Promoter Score (NPS)', 0, 10, 5, 
                        help="The Net Promoter Score (NPS) measures customer loyalty and satisfaction on a scale from 0 to 10.")

purchase_value = st.sidebar.slider('Purchase Value (USD)', 0, 1000, 100, 
                                   help="The total value of the customer’s purchases in USD.")

website_page_views = st.sidebar.slider('Website Page Views', 0, 100, 20, 
                                       help="The number of web pages viewed by the customer.")

website_time_spent = st.sidebar.slider('Website Time Spent (in minutes)', 0, 500, 60, 
                                       help="The total time the customer spent on the website in minutes.")

marketing_communication_sent_open_diff = st.sidebar.slider('Marketing Communication Sent vs Open Difference', 0, 20, 5, 
                                                           help="Difference between the number of marketing emails sent and opened by the customer.")

service_interactions_call = st.sidebar.slider('Service Interactions - Call', 0, 10, 2, 
                                              help="The number of service interactions (e.g., customer support calls).")

service_interactions_email = st.sidebar.slider('Service Interactions - Email', 0, 10, 2, 
                                               help="The number of email interactions the customer had with the service team.")

service_interactions_chat = st.sidebar.slider('Service Interactions - Chat', 0, 10, 2, 
                                              help="The number of chat interactions the customer had with the service team.")

payment_history_late_payments = st.sidebar.slider('Number of Late Payments', 0, 10, 1, 
                                                  help="The number of late payments made by the customer.")

subscription_duration = st.sidebar.slider('Subscription Duration (in months)', 0, 60, 12, 
                                          help="Total duration of the customer’s subscription.")

# Real-time charts
st.write("### Customer Engagement Breakdown")
engagement_data = {'Metric': ['NPS', 'Page Views', 'Time Spent', 'Late Payments'],
                   'Value': [nps, website_page_views, website_time_spent, payment_history_late_payments]}

engagement_df = pd.DataFrame(engagement_data)

# Bar chart of engagement metrics
st.bar_chart(engagement_df.set_index('Metric'))

# Create a button to predict churn
if st.button('Predict Churn'):
    # Create a DataFrame with user input
    user_data = pd.DataFrame({
        'NPS': [nps],
        'PurchaseValue': [purchase_value],
        'WebsitePageViews': [website_page_views],
        'WebsiteTimeSpent': [website_time_spent],
        'MarketingCommunicationSentOpenDiff': [marketing_communication_sent_open_diff],
        'ServiceInteractions_Call': [service_interactions_call],
        'ServiceInteractions_Email': [service_interactions_email],
        'ServiceInteractions_Chat': [service_interactions_chat],
        'PaymentHistoryNoOfLatePayments': [payment_history_late_payments],
        'SubscriptionDuration': [subscription_duration]
    })

    # Make prediction using the loaded model
    prediction = loaded_model.predict(user_data)

    # Display the prediction with interactive feedback
    if prediction[0] == 1:
        st.error('⚠️ The customer is predicted to churn. Take preventive action!')
    else:
        st.success('🎉 The customer is predicted not to churn. They are likely to stay.')

    # Display real-time engagement insights
    st.write("### Real-Time Insights Based on Your Inputs")
    
    fig, ax = plt.subplots()
    ax.bar(engagement_df['Metric'], engagement_df['Value'], color='skyblue')
    ax.set_ylabel('Value')
    ax.set_title('Customer Engagement Metrics')
    st.pyplot(fig)

# Footer with extra tips
st.write("""
---
💡 **Tip:** Use the sliders on the sidebar to adjust metrics and observe how they impact the predicted churn risk.
""")
