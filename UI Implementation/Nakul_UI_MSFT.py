import streamlit as st
import pandas as pd

st.title("Stock Market prediction models")



# Data for the table
data = {
    "Serial Number": [1, 2, 3],
    "Name of Model": ["Nakul_MSFT", "Apple_Navya", "Nishith_ITC"],
    "Pre Trained Model":['<a href="https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/main/PreTrainedModel/lstm_model.h5" target="_blank">model</a>',
                         '<a href="https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/main/PreTrainedModel/lstm_model.h5" target="_blank">model</a>',
                         '<a href="https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/main/PreTrainedModel/lstm_model.h5" target="_blank">model</a>'],
    "Model code": ['<a href="https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/main/Nakul_MSFT.ipynb" target="_blank">code</a>', 
                   '<a href="https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/main/Apple_Stock_Price_Prediction(Navya).ipynb" target="_blank">code</a>', 
                   '<a href="https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/main/Nishith_ITC.ipynb" target="_blank">code</a>'],
    "Dataset Name": ["Microsoft", "Apple", "ITC"]
}

# Create a DataFrame
df = pd.DataFrame(data)

df_html = df.to_html(escape=False)
st.write(df_html, unsafe_allow_html=True)

evaluation_metrics = {
    "Nakul_MSFT": {"Root Mean Squared Error (RMSE)  " :  0.019200205320298192, "Mean Squared Error (MSE) ": 0.000368647884341607},
    "Apple_Navya": {"Root Mean Squared Error (RMSE)  " :  0.04512147790902757, "Mean Squared Error (MSE) " : 0.0020359477686948627},
    "Nishith_ITC": {"Root Mean Squared Error (RMSE) " :  0.02456554857223372, "Mean Squared Error (MSE) " : 0.0006034661766547741}
    # Add your own models and their metrics here
}
model_descriptions = {
    "Nakul_MSFT": "Nakul_MSFT is a model trained on the Microsoft stock price data using LSTM architecture. It aims to predict the future stock prices based on historical data.",
    "Apple_Navya": "Apple_Navya is a model designed for predicting Apple's stock prices. It leverages an LSTM network to capture the temporal dependencies in the stock price data.",
    "Nishith_ITC": "Nishith_ITC is a predictive model for ITC stock prices. It utilizes an LSTM network to forecast future prices based on past trends."
}

def main():
    st.title("Model Evaluation Metrics")  

    # Model selection
  
    selected_model = st.selectbox('Select a model',['Nakul_MSFT', 'Apple_Navya', 'Nishith_ITC'])

    # Display evaluation metrics for the selected model
    if selected_model:
        st.subheader(f"Evaluation Metrics for {selected_model}")
        metrics = evaluation_metrics[selected_model]
        for metric_name, metric_value in metrics.items():
            st.write(f"{metric_name}: {metric_value}")
    
    st.subheader(f"Description of {selected_model}")
    st.write(model_descriptions[selected_model])
    
    model_ratings = {model: [] for model in evaluation_metrics}
    st.subheader(f"Rate {selected_model}")
    rating = st.slider("Rate this model (1 to 5)", 1, 5)
        
    if st.button("Submit Rating"):
        model_ratings[selected_model].append(rating)
        st.success(f"Rating submitted: {rating}")

        # Display average rating
    if model_ratings[selected_model]:
        average_rating = sum(model_ratings[selected_model]) / len(model_ratings[selected_model])
        st.write(f"Average Rating for {selected_model}: {average_rating:.2f}")

if __name__ == "__main__":
    main()

st.sidebar.title("Sidebar")
selected_functionality = st.sidebar.selectbox(
        "Select Functionality",
        ["Home", "Compare Models", "Model Description", "Upload Data"]
    )

# Section for adding a new model
st.header('Add a New Model')

# Input for the new model name
model_name = st.text_input('Enter the name of model')

# GitHub link for adding the model
github_link = "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25"


if model_name:
    st.write(f'please visit the following GitHub repository to add a new model:')

    st.markdown(f'''
        <a href="{github_link}" target="_blank">
            <button style="background-color: #4CAF50; border: none; color: white; padding: 15px 32px;
            text-align: center; text-decoration: none; display: inline-block; font-size: 16px; margin: 4px 2px;
            cursor: pointer;">Go to GitHub Repository</button>
        </a>
    ''', unsafe_allow_html=True)
st.header('Help us Improve')

name = st.text_input("Enter your name: ", "")

# Date Input
date = st.date_input("Select a date: ")

# Text Area
feedback = st.text_area("Feedback:", "")
if st.button("Submit Details"):
        st.success(f"Submitted Successfully!!")
