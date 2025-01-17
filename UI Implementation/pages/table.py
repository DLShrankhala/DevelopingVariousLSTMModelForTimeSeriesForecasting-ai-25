import streamlit as st
import pandas as pd
import os

# Function to load or create initial data
def load_data(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        return pd.DataFrame({
            "Serial Number": [],
            "Model Name": [],
            "Pre-trained Model": [],
            "Model Code": [],
            "Dataset Used": [],
            "Link to Dataset": []
        })

# File path for saving and loading data
DATA_FILE = "model_data.csv"

# Initial model data
initial_model_data = [
    {
        "Serial Number": 1,
        "Model Name": "Apple Company Stock Price Prediction using LSTM",
        "Pre-trained Model": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/PreTrainedModel/trained_model.pkl",
        "Model Code": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/Apple_Stock_Price_Prediction(Navya).ipynb",
        "Dataset Used": "Apple Stock Dataset",
        "Link to Dataset": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/Data/AAPL.csv"
    },
    {
        "Serial Number": 2,
        "Model Name": "Microsoft Stock Price Prediction using LSTM",
        "Pre-trained Model": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/PreTrainedModel/trained_model.pkl",
        "Model Code": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/Nakul_MSFT.ipynb",
        "Dataset Used": "Microsoft Stock Dataset",
        "Link to Dataset": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/Data/MSFT.csv"
    },
    {
        "Serial Number": 3,
        "Model Name": "Adani Ports Stock Price Prediction using LSTM",
        "Pre-trained Model": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/PreTrainedModel/trained_model.pkl",
        "Model Code": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/Siddhant_AdaniPorts_LSTM.ipynb",
        "Dataset Used": "Adani Stock Dataset",
        "Link to Dataset": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/Data/ADANIPORTS.csv"
    },
    {
        "Serial Number": 4,
        "Model Name": "5Paisa Stock Price Prediction using LSTM",
        "Pre-trained Model": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/PreTrainedModel/trained_model.pkl",
        "Model Code": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/Aryan_5PAISA_LSTM.ipynb",
        "Dataset Used": "5Paisa Stock Dataset",
        "Link to Dataset": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/Data/5PAISA.csv"
    },
    {
        "Serial Number": 5,
        "Model Name": "SBI Stock Price Prediction using LSTM",
        "Pre-trained Model": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/PreTrainedModel/trained_model.pkl",
        "Model Code": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/SBI_Stock_Prediction%20(Surya).ipynb",
        "Dataset Used": "SBI Stock Dataset",
        "Link to Dataset": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/Data/SBIN.csv"
    },
    {
        "Serial Number": 6,
        "Model Name": "ITC Stock Price Prediction using LSTM",
        "Pre-trained Model": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/PreTrainedModel/trained_model.pkl",
        "Model Code": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/Nishith_ITC.ipynb",
        "Dataset Used": "ITC Stock Dataset",
        "Link to Dataset": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/Data/ITC.csv"
    },
    {
        "Serial Number": 7,
        "Model Name": "TCS Stock Price Prediction using LSTM",
        "Pre-trained Model": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/PreTrainedModel/trained_model.pkl",
        "Model Code": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/shruti.ipynb",
        "Dataset Used": "TCS Stock Dataset",
        "Link to Dataset": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/Data/TATAMOTORS.NS.csv"
    },
    {
        "Serial Number": 8,
        "Model Name": "BAJAJ Stock Price Prediction using LSTM",
        "Pre-trained Model": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/PreTrainedModel/trained_model.pkl",
        "Model Code": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/Saumya_LSTM(loaded)_BAJAJ.ipynb",
        "Dataset Used": "BAJAJ Stock Dataset",
        "Link to Dataset": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/Data/BAJAJ-AUTO.csv"
    },
    {
        "Serial Number": 9,
        "Model Name": "RELIANCE Stock Price Prediction using LSTM",
        "Pre-trained Model": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/PreTrainedModel/trained_model.pkl",
        "Model Code": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/Validation_reliance_shuryansh.ipynb",
        "Dataset Used": "RELIANCE Stock Dataset",
        "Link to Dataset": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/Data/RELIANCE.NS.csv"
    },
    {
        "Serial Number": 10,
        "Model Name": "TESLA Stock Price Prediction using LSTM",
        "Pre-trained Model": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/PreTrainedModel/trained_model.pkl",
        "Model Code": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/harshitha_train_%26_load_model_TSLA_.ipynb",
        "Dataset Used": "TESLA Stock Dataset",
        "Link to Dataset": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/Data/TSLA%20(1).csv"
    },
    {
        "Serial Number": 11,
        "Model Name": "MERCEDEZ Stock Price Prediction using LSTM",
        "Pre-trained Model": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/PreTrainedModel/trained_model.pkl",
        "Model Code": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/model_loaded_train_tathagata_MBG.ipynb",
        "Dataset Used": "Mercedez Stock Dataset",
        "Link to Dataset": "https://www.kaggle.com/datasets/mysarahmadbhat/mercedes-used-car-listing"
    },
    {
        "Serial Number": 12,
        "Model Name": "BSE_SENSEX Stock Price Prediction using LSTM",
        "Pre-trained Model": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/PreTrainedModel/trained_model.pkl",
        "Model Code": "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25/blob/dfe0b7b164f97a92c469c33cfe59a78c0a88c6c4/BSE_SENSEX_Niraj.ipynb",
        "Dataset Used": "BSE Sensex Stock Dataset",
        "Link to Dataset": "https://www.kaggle.com/datasets/maheshmani13/bse-sensex-10-year-stock-price"
    }
]

# Load existing data or create an empty DataFrame
df = load_data(DATA_FILE)

# If DataFrame is empty, initialize it with initial model data
if df.empty:
    df = pd.DataFrame(initial_model_data)

# Function to convert the DataFrame to a clickable link with placeholder
def make_clickable(val, placeholder="Yes"):
    if isinstance(val, str) and val.startswith("http"):
        return f'<a href="{val}" target="_blank">{placeholder}</a>'
    return val

# Apply the make_clickable function to the 'Pre-trained Model', 'Modeling Script', and 'Link to Dataset' columns
df['Pre-trained Model'] = df['Pre-trained Model'].apply(lambda x: make_clickable(x, "Link"))
df['Model Code'] = df['Model Code'].apply(lambda x: make_clickable(x, "Link"))
df['Link to Dataset'] = df['Link to Dataset'].apply(lambda x: make_clickable(x, "Link"))

st.title("Stock Price Prediction Models")

# Display the table with clickable links
st.write("### Model Information")
st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

# Provide a way to add more models
st.write("### Add a New Model")
model_name = st.text_input("Model Name")
pretrained_model_link = st.text_input("Link to Pre-trained Model")
modeling_script_link = st.text_input("Link to Model Code")
dataset_used = st.text_input("Dataset Used")
link_to_dataset = st.text_input("Link to Dataset")

if st.button("Add Model"):
    if model_name and pretrained_model_link and modeling_script_link and dataset_used and link_to_dataset:
        new_model = pd.DataFrame([{
            "Serial Number": len(df) + 1,
            "Model Name": model_name,
            "Pre-trained Model": pretrained_model_link,
            "Model Code": modeling_script_link,
            "Dataset Used": dataset_used,
            "Link to Dataset": link_to_dataset
        }])
        new_model['Pre-trained Model'] = new_model['Pre-trained Model'].apply(lambda x: make_clickable(x, "Link"))
        new_model['Model Code'] = new_model['Model Code'].apply(lambda x: make_clickable(x, "Link"))
        new_model['Link to Dataset'] = new_model['Link to Dataset'].apply(lambda x: make_clickable(x, "Link"))
        df = pd.concat([df, new_model], ignore_index=True)
        df.to_csv(DATA_FILE, index=False)  # Save updated data to file
        st.success("Model added successfully!")
    else:
        st.error("Please fill in all fields.")

# Display the updated table
st.write("### Updated Model Information")
st.markdown(df.to_html(escape=False, index=False), unsafe_allow_html=True)

# Provide a link to contribute to the GitHub repository
st.write("### Contribute to the GitHub Repository")
contribute_url = "https://github.com/DLShrankhala/DevelopingVariousLSTMModelForTimeSeriesForecasting-ai-25"
st.markdown(f"[Contribute to the GitHub Repository]({contribute_url})")