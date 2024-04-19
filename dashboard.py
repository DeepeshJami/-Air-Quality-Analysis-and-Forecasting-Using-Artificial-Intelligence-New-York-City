from numpy import number
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

# Load the data with cache
def load_original_data():
    data = pd.read_csv('Air_Quality.csv')
    data['Start_Date'] = pd.to_datetime(data['Start_Date'])
    data['Year'] = data['Start_Date'].dt.year
    return data

def load_filtered_data():
    data2 = pd.read_csv('Filtered_Nitrogen_Dioxide_Data.csv')
    data2['Start_Date'] = pd.to_datetime(data2['Start_Date'])
    data2.set_index('Start_Date', inplace=True)
    return data2

data = load_original_data()
data2 = load_filtered_data()

st.title('Air Quality Visualization Dashboard')
dataset_choice = st.sidebar.selectbox("Choose the dataset", ["Original Air Quality", "Filtered Nitrogen Dioxide"])

if dataset_choice == "Original Air Quality":
    st.header("Distribution of Data Values")
    fig, ax = plt.subplots(figsize=(12, 6))  # Adjusted size
    sns.histplot(data['Data Value'], kde=True, bins=30, ax=ax, color='skyblue')
    ax.set_title('Distribution of Data Values', fontsize=16)
    ax.set_xlabel('Data Value', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    st.pyplot(fig)

    st.header("Average Data Value by Year")
    fig, ax = plt.subplots(figsize=(12, 6))
    yearly_trend = data.groupby('Year')['Data Value'].mean()
    yearly_trend.plot(kind='line', marker='o', ax=ax, color='green')
    ax.set_title('Average Data Value by Year', fontsize=16)
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Average Data Value', fontsize=14)
    st.pyplot(fig)

    st.header("Average Data Value by Geographic Location")
    fig, ax = plt.subplots(figsize=(14, 8))
    geo_trend = data.groupby('Geo Place Name')['Data Value'].mean().sort_values(ascending=False)
    geo_trend.plot(kind='bar', ax=ax, color='purple')
    ax.set_title('Average Data Value by Geographic Location', fontsize=16)
    ax.set_xlabel('Geographic Location', fontsize=14)
    ax.set_ylabel('Average Data Value', fontsize=14)
    ax.tick_params(axis='x', rotation=90)
    st.pyplot(fig)

    st.header("Data Value Distribution by Indicator ID")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.boxplot(x='Indicator ID', y='Data Value', data=data, ax=ax)
    ax.set_title('Data Value Distribution by Indicator ID', fontsize=16)
    ax.set_xlabel('Indicator ID', fontsize=14)
    ax.set_ylabel('Data Value', fontsize=14)
    st.pyplot(fig)

elif dataset_choice == "Filtered Nitrogen Dioxide":
    st.header("Distribution of Data Values")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(data2['Data Value'], kde=True, bins=30, ax=ax, color='blue')
    ax.set_title('Distribution of Data Values', fontsize=16)
    ax.set_xlabel('Data Value', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    st.pyplot(fig)

    st.header("Time Series Decomposition")
    if len(data2) > 365:  # Ensure there is enough data for decomposition
        result = seasonal_decompose(data2['Data Value'], model='additive', period=365)
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 23))
        result.trend.plot(ax=ax1)
        ax1.set_title('Trend', fontsize=14)
        ax1.set_ylabel('Trend', fontsize=12)
        result.seasonal.plot(ax=ax2)
        ax2.set_title('Seasonality', fontsize=14)
        ax2.set_ylabel('Seasonality', fontsize=12)
        result.resid.plot(ax=ax3)
        ax3.set_title('Residuals', fontsize=14)
        ax3.set_ylabel('Residuals', fontsize=12)
        st.pyplot(fig)
    else:
        st.write("Data2 needs at least one full year of data for decomposition.")

    def load_filtered_data():
        data2 = pd.read_csv('Filtered_Nitrogen_Dioxide_Data.csv')
        data2['Start_Date'] = pd.to_datetime(data2['Start_Date'])
        data2.set_index('Start_Date', inplace=True)
        # Convert all columns that should be numeric but are object type due to parsing errors
        for col in data2.columns:
            if data2[col].dtype == 'object':
                try:
                    data2[col] = pd.to_numeric(data2[col])
                except ValueError:
                    pass  # Or handle/log the columns that cannot be converted
        return data2

    data2 = load_filtered_data()

    st.header("Correlation Heatmap")
    # Select only numeric columns for correlation matrix
    numeric_cols = data2.select_dtypes(include=[number])
    if len(numeric_cols.columns) > 1:
        fig, ax = plt.subplots(figsize=(10, 8))
        corr_matrix = numeric_cols.corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        ax.set_title('Correlation Heatmap', fontsize=16)
        st.pyplot(fig)
    else:
        st.write("Not enough numeric features to display a correlation heatmap.")

    st.header("30-Day Rolling Average of Data Values")
    fig, ax = plt.subplots(figsize=(20, 6))
    data2['Data Value'].rolling(window=30).mean().plot(ax=ax, color='magenta')
    ax.set_title('30-Day Rolling Average of NO2 Levels', fontsize=16)
    ax.set_xlabel('Date', fontsize=14)
    ax.set_ylabel('NO2 Levels', fontsize=14)
    st.pyplot(fig)
