# %%
import pandas as pd
import numpy as np
from faker import Faker
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st


df = pd.read_csv("dam_water_prediction_dataset_with_outliers.csv")


# Title for the app
st.title("Dam Dataset Overview")

# Sidebar Buttons with Highlighting Logic
selected_button = st.sidebar.selectbox("Go to", ["Dam Dataset Overview", "EDA", "Numerical Data"])

# Define a custom CSS to highlight the selected button
highlight_css = """
    <style>
        .highlighted-button {
            background-color: #007bff;
            color: white;
        }
    </style>
"""

# Inject custom CSS into the app
st.markdown(highlight_css, unsafe_allow_html=True)

# Display Content Based on Selected Button
if selected_button == "Dam Dataset Overview":
    st.markdown("# Dam Health Prediction Dataset")  

    st.dataframe(df)


    st.title("Dam Dataset Feature Explanation")


    features = """
    ### Dam Dataset Features

    - **Dam_ID**:  
        - A unique identifier for each dam.  
        - Generated as a universally unique identifier (UUID). <br>
    <br>
    - **Dam_Health**:  
        - The condition or health status of the dam.  
        - Categorical values: `Good`, `Average`, and `Poor`.  
        - Probability distribution: 60% `Good`, 30% `Average`, 10% `Poor`. <br>
    <br>
    - **Water_Level_m**:  
        - The current water level in the dam, measured in meters.  
        - Generated using a normal distribution with a mean of 100 meters and a standard deviation of 30 meters.  
        - Includes outliers (extremely high values). <br>
    <br>
    - **Water_Inflow_cms**:  
        - The rate of water entering the dam, measured in cubic meters per second (cms).  
        - Normally distributed with a mean of 500 cms and a standard deviation of 150 cms.  
        - Includes outliers (extremely high values). <br>
    <br>
    - **Water_Outflow_cms**:  
        - The rate of water leaving the dam, measured in cubic meters per second (cms).  
        - Normally distributed with a mean of 480 cms and a standard deviation of 145 cms.  
        - Includes outliers (extremely low values). <br>
    <br>
    - **Reservoir_Capacity_percent**:  
        - The percentage of the reservoir’s capacity currently utilized.  
        - Randomly generated values between 30% and 100%. <br>
    <br>
    - **Sedimentation_Rate_m_per_year**:  
        - The rate at which sediment accumulates in the dam, measured in meters per year.  
        - Normally distributed with a mean of 0.5 meters and a standard deviation of 0.2 meters.  
        - Includes outliers (extremely high values). <br>
    <br>
    - **Water_Quality_Index**:  
        - A measure of the overall quality of water in the dam, ranging from 50 (poor) to 100 (excellent).  
        - Uniformly distributed values.  
        - Includes outliers (extremely low values). <br>
    <br>
    - **Temperature_C**:  
        - Ambient temperature near the dam, measured in degrees Celsius.  
        - Randomly generated values between 15°C and 35°C. <br>
    <br>
    - **Humidity_percent**:  
        - The percentage of atmospheric moisture near the dam.  
        - Randomly generated values between 40% and 90%. <br>
    <br>
    - **Wind_Speed_kmh**:  
        - Wind speed near the dam, measured in kilometers per hour.  
        - Randomly generated values between 5 km/h and 50 km/h. <br>
    <br>
    - **Soil_Moisture_percent**:  
        - The moisture content of soil near the dam, represented as a percentage.  
        - Randomly generated values between 10% and 50%. <br>
    <br>
    - **Evaporation_Rate_mm_per_day**:  
        - The rate of water evaporation from the dam’s surface, measured in millimeters per day.  
        - Randomly generated values between 2 mm/day and 10 mm/day. <br>
    <br>
    - **Ponding_mm**:  
        - The depth of water ponding near the dam, measured in millimeters.  
        - Randomly generated values between 0 mm and 5 mm.<br>
        <br>
    - **Water_Temperature_C**:  
        - The temperature of the water in the dam, measured in degrees Celsius.  
        - Randomly generated values between 15°C and 30°C.
    """

    st.markdown("""
    ### Dataset Features
    - **Dam_ID**: Unique identifier for the dam.
    - **Dam_Health**: Health status of the dam.
    - **Water_Level_m**: Current water level (in meters).
    - **Reservoir_Capacity_percent**: Percentage capacity of the reservoir.
    - **Water_Quality_Index**: Quality index of water (50 to 100).
    """)
    st.markdown(features, unsafe_allow_html=True)

    Numerical_Values = """
       <h1>Numerical Features</h1>
<ul>
    <li><strong>Water_Level_m</strong></li>
    <li><strong>Water_Inflow_cms</strong></li>
    <li><strong>Water_Outflow_cms</strong></li>
    <li><strong>Reservoir_Capacity_percent</strong></li>
    <li><strong>Sedimentation_Rate_m_per_year</strong></li>
    <li><strong>Temperature_C</strong></li>
    <li><strong>Humidity_percent</strong></li>
    <li><strong>Wind_Speed_kmh</strong></li>
    <li><strong>Evaporation_Rate_mm_per_day</strong></li>
</ul>

"""
    categorical_valued ="""
<h1>Categorical Features</h1>
<ul>
    <li><strong>Reservoir_Type</strong></li>
    <li><strong>Region</strong></li>
    <li><strong>Dam_Type</strong></li>
    <li><strong>Season</strong></li>
</ul>

"""
    st.markdown(Numerical_Values,unsafe_allow_html=True)
    st.markdown(categorical_valued,unsafe_allow_html=True)
    st.markdown("<h1>Overview of Dataset</h1>",unsafe_allow_html=True)
    info_dict = {
    "Column": df.columns,
    "Non-Null Count": df.notnull().sum(),
    "Dtype": df.dtypes
    }

    df_info = pd.DataFrame(info_dict)

 
    st.table(df_info)


elif selected_button == "EDA":
    st.subheader("EDA")
    
    # Identify Categorical Columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        st.write(f"Categorical Columns: {', '.join(categorical_cols)}")
        for col in categorical_cols:
            if col == "Dam_Name":
                continue
            st.markdown(f"### {col}")
            value_counts = df[col].value_counts()
            st.bar_chart(value_counts)
            st.write(f"Table of values in {col}:")
            st.table(value_counts.reset_index().rename(columns={col: "Count", "index": col}))
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if numerical_cols:
        for col in numerical_cols:
            st.markdown(f"### {col}")
            st.line_chart(df[col])
            st.write(f"Summary Statistics for {col}:")
            st.table(df[col].describe())
    else:
        st.write("No categorical columns found in the dataset.")
    st.markdown("<h1>Missing Value Visualization </h1>",unsafe_allow_html=True)
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')

    # Display the heatmap using st.pyplot()
    st.pyplot(plt)
elif selected_button == "Numerical Data":
    st.subheader("Numerical Data Overview")
    
    # Identify Numerical Columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    
    if numerical_cols:
        st.write(f"Numerical Columns: {', '.join(numerical_cols)}")
        for col in numerical_cols:
            st.markdown(f"### {col}")
            st.line_chart(df[col])
            st.write(f"Summary Statistics for {col}:")
            st.table(df[col].describe())
    else:
        st.write("No numerical columns found in the dataset.")

