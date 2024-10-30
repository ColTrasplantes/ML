# Install necessary libraries
!pip install pandas numpy scikit-survival openpyxl

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored

# Set random seed for reproducibility
np.random.seed(42)

# Number of observations
n = 10000

# Generate covariates
age = np.random.normal(50, 10, n)  # Age around 50 years
bmi = np.random.normal(25, 5, n)  # BMI around 25
diabetes = np.random.binomial(1, 0.2, n)  # 20% have diabetes
hypertension = np.random.binomial(1, 0.3, n)  # 30% have hypertension
donor_age = np.random.binomial(1, 0.4, n)  # 40% donors older than 60
cold_ischemia_time = np.random.normal(10, 2, n)  # Cold ischemia time around 10 hours

# Simulate time to event (graft loss) and event occurrence
baseline_hazard = 0.01
hazard_ratio = np.exp(0.05 * age + 0.1 * bmi + 1.0 * diabetes + 0.8 * hypertension + 1.2 * donor_age + 0.08 * cold_ischemia_time)
time_to_event = np.random.exponential(1 / (baseline_hazard * hazard_ratio))
event_occurred = np.random.binomial(1, 0.5, n)  # 50% event occurrence

# Censor time to event at 60 months
time_to_event = np.minimum(time_to_event, 60)

# Create DataFrame
data = pd.DataFrame({
    'age': age,
    'bmi': bmi,
    'diabetes': diabetes,
    'hypertension': hypertension,
    'donor_age': donor_age,
    'cold_ischemia_time': cold_ischemia_time,
    'time_to_event': time_to_event,
    'event_occurred': event_occurred
})

# Split data into training and testing sets
X = data.drop(columns=['time_to_event', 'event_occurred'])
y = np.array([(event_occurred[i], time_to_event[i]) for i in range(len(event_occurred))], dtype=[('event', '?'), ('time', '<f8')])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Fit the Random Survival Forest model
rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, max_features="sqrt", n_jobs=-1, random_state=42)
rsf.fit(X_train, y_train)

# Predict on the test set
y_pred = rsf.predict(X_test)
c_index = concordance_index_censored(y_test['event'], y_test['time'], y_pred)
print(f'Concordance Index: {c_index[0]}')

# Save the dataset to an Excel file
data.to_excel('transplant_database.xlsx', index=False)
