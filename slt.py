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
selected_button = st.sidebar.selectbox("Go to", ["Dam Dataset Overview", "EDA", "Preprocessing","Logistic Regression Model"])

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
    st.markdown("<h1>Missing Values in Each Column</h1>",unsafe_allow_html=True)
    missing_values = df.isnull().sum()
    st.write(missing_values)
    df = df.drop("Dam_name",axis=1)
elif selected_button == "Preprocessing":
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.impute import SimpleImputer


    # Show subheader for Preprocessing
    st.subheader("Preprocessing")

    # Define categorical and numerical columns
    categorical_columns = ['Dam_Name', 'Dam_Health', 'Soil_Type', 'Region', 'Water_Quality', 'Water_Usage']
    numerical_columns = [
        'Water_Level_m', 'Water_Inflow_cms', 'Water_Outflow_cms',
        'Reservoir_Capacity_percent', 'Sedimentation_Rate_m_per_year','Temperature_C', 'Humidity_percent',
        'Wind_Speed_kmh', 'Evaporation_Rate_mm_per_day'
    ]

    # Visualize missing data before imputation
    st.write("### Missing Data Before Imputation")
    text = """
In this preprocessing step, we are handling missing values using two types of imputation techniques:

**Frequency Imputation** is used for categorical columns like 'Dam_Name', 'Dam_Health', 'Soil_Type', 'Region', 'Water_Quality', and 'Water_Usage'. This technique replaces missing values with the most frequent value (mode) of the column. It ensures that the imputation reflects the most common occurrence in the dataset, thus preserving the categorical distribution.

**Median Imputation** is applied to numerical columns such as 'Water_Level_m', 'Water_Inflow_cms', 'Water_Outflow_cms', and others. This method replaces missing values with the median of the column, which is robust to outliers and better preserves the central tendency of the data. It avoids the skewing effect that might occur with mean imputation due to extreme values.
"""

    st.markdown(text)
    missing_before = df.isnull().sum()
    st.bar_chart(missing_before)

    # Apply SimpleImputer to handle missing values
    categorical_imputer = SimpleImputer(strategy='most_frequent')
    df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])

    numerical_imputer = SimpleImputer(strategy='median')
    df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])

    # Visualize missing data after imputation
    st.write("### Missing Data After Imputation")
    missing_after = df.isnull().sum()
    st.bar_chart(missing_after)

    # Optional: Display a comparison plot
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # Plot missing data before imputation
    sns.barplot(x=missing_before.index, y=missing_before.values, ax=ax[0])
    ax[0].set_title('Missing Data Before Imputation')
    ax[0].set_ylabel('Number of Missing Values')
    ax[0].set_xlabel('Columns')
    ax[0].tick_params(axis='x', rotation=90)

    # Plot missing data after imputation
    sns.barplot(x=missing_after.index, y=missing_after.values, ax=ax[1])
    ax[1].set_title('Missing Data After Imputation')
    ax[1].set_ylabel('Number of Missing Values')
    ax[1].set_xlabel('Columns')
    ax[1].tick_params(axis='x', rotation=90)

    # Display comparison plot in Streamlit
    st.pyplot(fig)
    # Create a pair plot
    st.text("# PairPlot")
    sns.pairplot(df[df.columns], hue='Dam_Health')
    st.pyplot()

    # Visualize box plots before outlier removal
    st.text("# BoxPlots (Before Outlier Removal)")
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(3, 4, i)  # 3 rows, 4 columns of subplots
        sns.boxplot(data=df, x=col)
        plt.title(f"Box plot of {col}")
        plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot()

    # Calculate IQR and remove outliers
    Q1 = df.select_dtypes(include=[np.number]).quantile(0.25)
    Q3 = df.select_dtypes(include=[np.number]).quantile(0.75)
    IQR = Q3 - Q1
    df = df[~((df.select_dtypes(include=[np.number]) < (Q1 - 1.5 * IQR)) | 
                    (df.select_dtypes(include=[np.number]) > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Visualize box plots after outlier removal
    st.text("# BoxPlots (After Outlier Removal)")
    plt.figure(figsize=(12, 8))
    for i, col in enumerate(numeric_cols, 1):
        plt.subplot(3, 4, i)  # 3 rows, 4 columns of subplots
        sns.boxplot(data=df, x=col)
        plt.title(f"Box plot of {col}")
        plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot()
    st.markdown("<h1> Numeric Feature Distribution</h1>",unsafe_allow_html=True)



# Assuming you already have the 'df' DataFrame loaded

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    n = len(numeric_columns)
    cols = 3  
    rows = (n // cols) + (1 if n % cols != 0 else 0)  
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))  

    axes = axes.flatten()

    # Create histograms for each numeric column
    for i, column in enumerate(numeric_columns):
        axes[i].hist(df[column], bins=20, color='green', edgecolor='black')
        axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')

    # Turn off axes for any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout to avoid overlapping labels
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)

    text="""
    **Min-Max** scaling is used to normalize numerical data by rescaling the feature values to a specific range, usually [0, 1]. This helps to:

    - Improve model performance: Algorithms like gradient descent and k-nearest neighbors are sensitive to the scale of features, and normalization ensures that no feature dominates due to its larger range.
    - Ensure uniformity: It standardizes data when features have different units or scales, making it easier to compare and model them.
    This is particularly useful for algorithms that rely on distance measurements or those that are sensitive to feature magnitude.
    """
    st.markdown(text)
    from sklearn.preprocessing import StandardScaler

    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    scaler = StandardScaler()
    df[numeric_columns] = scaler.fit_transform(df[numeric_columns])
    # %%
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns

    n = len(numeric_columns)
    cols = 3  
    rows = (n // cols) + (1 if n % cols != 0 else 0)  
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))  

    axes = axes.flatten()

    # Create histograms for each numeric column
    for i, column in enumerate(numeric_columns):
        axes[i].hist(df[column], bins=20, color='green', edgecolor='black')
        axes[i].set_title(f'Histogram of {column}')
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Frequency')

    # Turn off axes for any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout to avoid overlapping labels
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)
    text= """
    <h2> Label Encoding</h2>
    For label encoding we are using ordinal encoder
    """

    st.markdown(text,unsafe_allow_html=True)
    # %%
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    info_dict = {
        'Column': df.columns,
        'Non-null count': df.notnull().sum(),
        'Dtype': df.dtypes
    }

    # Convert to DataFrame
    info_df = pd.DataFrame(info_dict)

    # Show it in Streamlit as a table
    st.markdown("<h3>Before Label Encoding</h3>",unsafe_allow_html=True)
    st.dataframe(info_df)
    cat_cols = []
    for col in df.columns:
        if df[col].dtype == 'object':
            cat_cols.append(col)

    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    info_dict = {
        'Column': df.columns,
        'Non-null count': df.notnull().sum(),
        'Dtype': df.dtypes
    }

    # Convert to DataFrame
    info_df = pd.DataFrame(info_dict)

    # Show it in Streamlit as a table
    st.markdown("<h3>After Label Encoding</h3>",unsafe_allow_html=True)
    st.dataframe(info_df)
    df.to_csv("dataset.csv", index=False)
elif selected_button == "Logistic Regression Model":
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Assuming df is already available

    st.subheader("Logistic Regression Model")
    df = pd.read_csv("dataset.csv")

    # Split data into features and target variable
 
   
    y = df['Dam_Health']
    X = df.drop('Dam_Health', axis=1)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize the Logistic Regression model
    model = LogisticRegression()

    # Train the model
    model.fit(X_train, y_train)

    # Get training and test scores
    training_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    # Display the scores in Streamlit
    st.markdown(f"<b>Training Score:</b> <h3>{training_score:.2f}</h3>",unsafe_allow_html=True)
    st.write(f"<b>Test Score:</b> <h3>{test_score:.2f}</h3>",unsafe_allow_html=True)

    # Display the classification report
    st.subheader("Classification Report")
    report = classification_report(y_test, model.predict(X_test), output_dict=True)
    st.write(report)

    # Display confusion matrix
    st.subheader("Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, model.predict(X_test))
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    st.pyplot(fig)

    # Optional: Visualize the training vs. test score comparison
    st.subheader("Model Performance Comparison")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=['Training', 'Test'], y=[training_score, test_score], ax=ax)
    plt.ylim(0, 1)
    st.pyplot(fig)
