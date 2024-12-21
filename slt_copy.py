# %%
import pandas as pd
import numpy as np
from faker import Faker
import seaborn as sns
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
fake = Faker()
Faker.seed(42)

# Define number of records
n_records = 3000

# Generate base features that will influence dam health
data = {
    "Dam_Name": [fake.city() + " Dam" for _ in range(n_records)],
    "Region": [fake.state() for _ in range(n_records)]
}

# Generate age factor (older dams tend to have more issues)
age_years = np.random.uniform(5, 100, size=n_records)

# Generate maintenance score (well-maintained dams are healthier)
maintenance_score = 100 - (age_years * 0.3) + np.random.normal(0, 10, size=n_records)
maintenance_score = np.clip(maintenance_score, 0, 100)

# Generate water level relative to capacity
water_level_ratio = np.random.normal(0.7, 0.15, size=n_records)
water_level_ratio = np.clip(water_level_ratio, 0.3, 1.0)

# Generate features with strong interdependencies
data.update({
    "Water_Level_m": 100 * water_level_ratio + np.random.normal(0, 5, size=n_records),
    "Water_Inflow_cms": np.random.normal(500, 50, size=n_records),
})

# Calculate outflow based on inflow and maintenance
data["Water_Outflow_cms"] = (
    data["Water_Inflow_cms"] * 
    (maintenance_score/100) * 0.95 + 
    np.random.normal(0, 10, size=n_records)
)

# Calculate reservoir capacity based on maintenance and age
data["Reservoir_Capacity_percent"] = (
    100 - (age_years * 0.2) + 
    (maintenance_score * 0.3) + 
    np.random.normal(0, 5, size=n_records)
)
data["Reservoir_Capacity_percent"] = np.clip(data["Reservoir_Capacity_percent"], 30, 100)

# Sedimentation rate increases with age and poor maintenance
data["Sedimentation_Rate_m_per_year"] = (
    (age_years * 0.01) - 
    (maintenance_score * 0.005) + 
    np.random.normal(0.2, 0.05, size=n_records)
)
data["Sedimentation_Rate_m_per_year"] = np.clip(data["Sedimentation_Rate_m_per_year"], 0, 2)

# Environmental factors
data.update({
    "Temperature_C": np.random.uniform(15, 35, size=n_records),
    "Humidity_percent": np.random.uniform(40, 90, size=n_records),
    "Wind_Speed_kmh": np.random.uniform(5, 50, size=n_records),
    "Soil_Type": np.random.choice(['Clay', 'Sandy', 'Loamy'], size=n_records),
    "Evaporation_Rate_mm_per_day": np.random.uniform(2, 10, size=n_records),
    "Water_Usage": np.random.choice(
        ['Agriculture', 'Domestic', 'Industrial'], 
        size=n_records, 
        p=[0.5, 0.3, 0.2]
    )
})

# Calculate water quality based on multiple factors
water_quality_score = (
    maintenance_score * 0.4 +
    (100 - age_years) * 0.3 +
    data["Reservoir_Capacity_percent"] * 0.3 +
    np.random.normal(0, 10, size=n_records)
)
data["Water_Quality"] = pd.cut(
    water_quality_score,
    bins=[-np.inf, 40, 70, np.inf],
    labels=['Poor', 'Good', 'Excellent']
)

# Calculate overall health score based on all factors
health_score = (
    maintenance_score * 0.3 +
    (100 - age_years) * 0.2 +
    data["Reservoir_Capacity_percent"] * 0.15 +
    (100 - data["Sedimentation_Rate_m_per_year"] * 50) * 0.15 +
    (water_quality_score) * 0.2 +
    np.random.normal(0, 5, size=n_records)
)
# Convert health score to categorical Dam_Health
data["Dam_Health"] = pd.cut(
    health_score,
    bins=[-np.inf, 60, 85, np.inf],  # Increased threshold for Poor category
    labels=['Poor', 'Average', 'Good']
)

# Create DataFrame
df = pd.DataFrame(data)

# Add some random nulls (5% of data)
for col in df.columns:
    mask = np.random.choice([True, False], size=n_records, p=[0.05, 0.95])
    df.loc[mask, col] = np.nan

# Add a few outliers (1% of data)
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    outlier_mask = np.random.choice([True, False], size=n_records, p=[0.01, 0.99])
    df.loc[outlier_mask, col] = df[col].max() * 1.5

# Create visualization of correlations
numeric_df = df.select_dtypes(include=[np.number])
numeric_df['Dam_Health_Code'] = pd.Categorical(df['Dam_Health']).codes

# Calculate and display correlations
plt.figure(figsize=(12, 8))
correlation_matrix = numeric_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Feature Correlations')
plt.tight_layout()
plt.show()

# Display Dam Health distribution
plt.figure(figsize=(8, 6))
df['Dam_Health'].value_counts().plot(kind='bar')
plt.title('Distribution of Dam Health')
plt.xlabel('Health Status')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Display first few rows and basic statistics
print("\nFirst few rows of the dataset:")
print(df.head())

print("\nBasic statistics of numeric features:")
print(df.describe())

print("\nCorrelations with Dam Health:")
correlations = numeric_df.corr()['Dam_Health_Code'].sort_values(ascending=False)
print(correlations)
sns.countplot(x='Dam_Health', data=df)
plt.show()

# %%

# %%
# Save the dataset to a CSV file
df.to_csv("dam_water_prediction_dataset_with_outliers.csv", index=False)

# Confirm that the dataset has been saved
print("Dataset saved successfully!")


# %%
df = pd.read_csv("dam_water_prediction_dataset_with_outliers.csv")

# %%


# %%
df.info()

# %%
df.duplicated().sum()

# %% [markdown]
# ##Data Cleaning
# 

# %%
df.isnull().sum()

# %%
df =df.dropna()

# %%
df.isnull().sum()

# %%
df.info()

# %%
import streamlit as st

import streamlit as st
import pandas as pd
import numpy as np


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

elif selected_button == "Categorical Data":
    st.subheader("Categorical Data Overview")
    
    # Identify Categorical Columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    if categorical_cols:
        st.write(f"Categorical Columns: {', '.join(categorical_cols)}")
        for col in categorical_cols:
            st.markdown(f"### {col}")
            value_counts = df[col].value_counts()
            st.bar_chart(value_counts)
            st.write(f"Table of values in {col}:")
            st.table(value_counts.reset_index().rename(columns={col: "Count", "index": col}))
    else:
        st.write("No categorical columns found in the dataset.")

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





# %%
cat_cols = []
for col in df.columns:
  if df[col].dtype == 'object':
    cat_cols.append(col)

# %%
for i in cat_cols:
   print(f"Unqiue value ${i} = {df[i].unique()}")

# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

y = df['Dam_Health']
X = df.drop('Dam_Health',axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
knn = LogisticRegression()
knn.fit(X_train,y_train)
training_score = knn.score(X_train,y_train)
test_score = knn.score(X_test,y_test)
print(f"Training score ${training_score} Test score ${test_score}")


# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
df = pd.read_csv("dam_water_prediction_dataset_with_outliers.csv")


# %%
cat_cols = []
for col in df.columns:
  if df[col].dtype == 'object':
    cat_cols.append(col)






# %%
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')

    



# %%
df = df.dropna(subset=['Dam_Name'], axis=0)



# %%
from sklearn.impute import SimpleImputer


categorical_columns = ['Dam_Name', 'Dam_Health', 'Soil_Type', 'Region','Water_Quality', 'Water_Usage']
numerical_columns = [
    'Water_Level_m', 'Water_Inflow_cms', 'Water_Outflow_cms',
    'Reservoir_Capacity_percent', 'Sedimentation_Rate_m_per_year','Temperature_C', 'Humidity_percent',
    'Wind_Speed_kmh', 'Evaporation_Rate_mm_per_day'
]



# Impute missing values for categorical columns using the most frequent strategy
categorical_imputer = SimpleImputer(strategy='most_frequent')
df[categorical_columns] = categorical_imputer.fit_transform(df[categorical_columns])
numerical_imputer = SimpleImputer(strategy='median')
df[numerical_columns] = numerical_imputer.fit_transform(df[numerical_columns])




# %%
import seaborn as sns
import matplotlib.pyplot as plt

numerical_cols = df.select_dtypes(include=[np.number]).columns
df[numerical_cols].hist(figsize=(20, 11), bins=20)



# %%
sns.pairplot(df[df.columns],hue='Dam_Health')


# %%
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Create a boxplot for each numeric column
numeric_cols
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 4, i)  # 3 rows, 4 columns of sub
    
    sns.boxplot(data=df, x=col)
    plt.title(f"Box plot of {col}")
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# %%
Q1 = df.select_dtypes(include=[np.number]).quantile(0.25)
Q3 = df.select_dtypes(include=[np.number]).quantile(0.75)
IQR = Q3 - Q1


# %%
df= df[~((df.select_dtypes(include=[np.number]) < (Q1 - 1.5 * IQR)) | 
                      (df.select_dtypes(include=[np.number]) > (Q3 + 1.5 * IQR))).any(axis=1)]

# %%
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Create a boxplot for each numeric column
numeric_cols
plt.figure(figsize=(12, 8))
for i, col in enumerate(numeric_cols, 1):
    plt.subplot(3, 4, i)  # 3 rows, 4 columns of subplots
    sns.boxplot(data=df, x=col)
    plt.title(f"Box plot of {col}")
    plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt


numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns


n = len(numeric_columns)
cols = 3  
rows = (n // cols) + (1 if n % cols != 0 else 0)  
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))  


axes = axes.flatten()


for i, column in enumerate(numeric_columns):
    axes[i].hist(df[column], bins=20, color='green', edgecolor='black')
    axes[i].set_title(f'Histogram of {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frequency')


for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()  
plt.show()


# %%
from sklearn.preprocessing import StandardScaler

numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
scaler = StandardScaler()
df[numeric_columns] = scaler.fit_transform(df[numeric_columns])

print(df.head())


# %%
import matplotlib.pyplot as plt


numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns


n = len(numeric_columns)
cols = 3  
rows = (n // cols) + (1 if n % cols != 0 else 0)  
fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))  


axes = axes.flatten()


for i, column in enumerate(numeric_columns):
    axes[i].hist(df[column], bins=20, color='green', edgecolor='black')
    axes[i].set_title(f'Histogram of {column}')
    axes[i].set_xlabel(column)
    axes[i].set_ylabel('Frequency')


for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()  
plt.show()


# %%
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.show()

# %%
sns.countplot(x='Dam_Health', data=df)
plt.show()


# %%
pd.crosstab(df['Water_Usage'], df['Dam_Health']).plot(kind='bar', stacked=True)
plt.show()


# %%
print(df['Dam_Health'].value_counts())


# %%
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in cat_cols:
    df[col] = le.fit_transform(df[col])

# %%
df.info()

# %%
# %%
y = df['Dam_Health']
X = df.drop('Dam_Health',axis=1)

# %%
X

# %%
X

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

y = df['Dam_Health']
X = df.drop('Dam_Health',axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
knn = LogisticRegression()
knn.fit(X_train,y_train)
training_score = knn.score(X_train,y_train)
test_score = knn.score(X_test,y_test)
print(f"Training score ${training_score} Test score ${test_score}")

# %%
