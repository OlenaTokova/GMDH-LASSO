import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Number of samples
num_samples = 1000

# Generate synthetic data
data = {
    'Austempering Temperature (°C)': np.random.uniform(200, 400, num_samples),
    'Austempering Time (minutes)': np.random.uniform(30, 180, num_samples),
    'Initial Temperature (°C)': np.random.uniform(800, 1200, num_samples),
    'Carbon (%)': np.random.uniform(0.1, 2.0, num_samples),
    'Silicon (%)': np.random.uniform(0.1, 2.0, num_samples),
    'Manganese (%)': np.random.uniform(0.1, 2.0, num_samples),
    'Chromium (%)': np.random.uniform(0.1, 1.5, num_samples),
    'Nickel (%)': np.random.uniform(0.1, 1.5, num_samples),
    'Molybdenum (%)': np.random.uniform(0.01, 0.5, num_samples),
    'Phosphorus (%)': np.random.uniform(0.01, 0.05, num_samples),
    'Sulfur (%)': np.random.uniform(0.01, 0.05, num_samples),
    'Titanium (%)': np.random.uniform(0.01, 0.1, num_samples),
    'Vanadium (%)': np.random.uniform(0.01, 0.1, num_samples),
    'Strength (MPa)': np.random.uniform(300, 1000, num_samples),
    'Melting Temperature (°C)': np.random.uniform(1200, 1600, num_samples),
    'Thermal Conductivity (W/m·K)': np.random.uniform(20, 50, num_samples),
    'Thermal Expansion Coefficient (1/K)': np.random.uniform(1e-6, 2e-5, num_samples),
    'Specific Heat (J/kg·K)': np.random.uniform(400, 700, num_samples),
    'Phase Transformation Temperature (°C)': np.random.uniform(600, 900, num_samples),
}

# Calculate Cooling Rate and Hardness based on synthetic relationships
data['Cooling Rate (°C/min)'] = (
    0.1 * data['Austempering Temperature (°C)'] +
    0.05 * data['Austempering Time (minutes)'] +
    0.01 * data['Initial Temperature (°C)'] +
    0.5 * data['Carbon (%)'] +
    0.3 * data['Silicon (%)'] +
    0.2 * data['Manganese (%)'] +
    0.1 * data['Chromium (%)'] +
    0.1 * data['Nickel (%)'] +
    0.05 * data['Molybdenum (%)'] +
    np.random.normal(0, 1, num_samples)
)

data['Hardness (HV)'] = (
    0.2 * data['Strength (MPa)'] +
    0.1 * data['Melting Temperature (°C)'] +
    0.05 * data['Thermal Conductivity (W/m·K)'] +
    0.01 * data['Thermal Expansion Coefficient (1/K)'] +
    0.1 * data['Specific Heat (J/kg·K)'] +
    0.2 * data['Phase Transformation Temperature (°C)'] +
    np.random.normal(0, 5, num_samples)
)

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('synthetic_alloy_data_extended.csv', index=False)
