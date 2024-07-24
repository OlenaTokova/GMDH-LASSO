import pandas as pd
import numpy as np

# Встановимо фіксований seed для відтворюваності
np.random.seed(42)

# Кількість зразків
num_samples = 1000

# Марки високопрочного чавуну та їх склад
ductile_iron_brands = {
    'ASTM_A536_60-40-18': {'Carbon (%)': 3.6, 'Silicon (%)': 2.4, 'Manganese (%)': 0.4, 'Magnesium (%)': 0.04, 'Phosphorus (%)': 0.05, 'Sulfur (%)': 0.01},
    'ASTM_A536_65-45-12': {'Carbon (%)': 3.7, 'Silicon (%)': 2.5, 'Manganese (%)': 0.4, 'Magnesium (%)': 0.04, 'Phosphorus (%)': 0.05, 'Sulfur (%)': 0.01},
    'ASTM_A536_80-55-06': {'Carbon (%)': 3.8, 'Silicon (%)': 2.6, 'Manganese (%)': 0.4, 'Magnesium (%)': 0.05, 'Phosphorus (%)': 0.05, 'Sulfur (%)': 0.01},
    'ASTM_A536_100-70-03': {'Carbon (%)': 3.9, 'Silicon (%)': 2.7, 'Manganese (%)': 0.4, 'Magnesium (%)': 0.05, 'Phosphorus (%)': 0.05, 'Sulfur (%)': 0.01},
    'ISO_1083_JS_400-18': {'Carbon (%)': 3.6, 'Silicon (%)': 2.3, 'Manganese (%)': 0.4, 'Magnesium (%)': 0.04, 'Phosphorus (%)': 0.04, 'Sulfur (%)': 0.02},
    'ISO_1083_JS_450-10': {'Carbon (%)': 3.7, 'Silicon (%)': 2.4, 'Manganese (%)': 0.4, 'Magnesium (%)': 0.04, 'Phosphorus (%)': 0.04, 'Sulfur (%)': 0.02},
    'ISO_1083_JS_500-7': {'Carbon (%)': 3.8, 'Silicon (%)': 2.5, 'Manganese (%)': 0.4, 'Magnesium (%)': 0.05, 'Phosphorus (%)': 0.04, 'Sulfur (%)': 0.02},
    'ISO_1083_JS_600-3': {'Carbon (%)': 3.9, 'Silicon (%)': 2.6, 'Manganese (%)': 0.4, 'Magnesium (%)': 0.05, 'Phosphorus (%)': 0.04, 'Sulfur (%)': 0.02},
    'EN-GJS-400-18': {'Carbon (%)': 3.5, 'Silicon (%)': 2.2, 'Manganese (%)': 0.3, 'Magnesium (%)': 0.04, 'Phosphorus (%)': 0.05, 'Sulfur (%)': 0.01},
    'EN-GJS-450-10': {'Carbon (%)': 3.6, 'Silicon (%)': 2.3, 'Manganese (%)': 0.3, 'Magnesium (%)': 0.04, 'Phosphorus (%)': 0.05, 'Sulfur (%)': 0.01},
    'EN-GJS-500-7': {'Carbon (%)': 3.7, 'Silicon (%)': 2.4, 'Manganese (%)': 0.3, 'Magnesium (%)': 0.04, 'Phosphorus (%)': 0.05, 'Sulfur (%)': 0.01},
    'EN-GJS-600-3': {'Carbon (%)': 3.8, 'Silicon (%)': 2.5, 'Manganese (%)': 0.3, 'Magnesium (%)': 0.05, 'Phosphorus (%)': 0.05, 'Sulfur (%)': 0.01}
}

# Згенеруємо синтетичні дані
data = {
    'Brand': np.random.choice(list(ductile_iron_brands.keys()), num_samples),
    'Tensile Strength (MPa)': np.random.uniform(400, 800, num_samples),
    'Hardness (HB)': np.random.uniform(200, 350, num_samples),
    'Elongation (%)': np.random.uniform(2.0, 18.0, num_samples),
    'Thermal Conductivity (W/m·K)': np.random.uniform(20, 40, num_samples),
    'Specific Heat (J/kg·K)': np.random.uniform(400, 700, num_samples),
    'Melting Temperature (°C)': np.random.uniform(1150, 1300, num_samples)
}

# Додамо дані про склад залежно від марки
composition_data = {key: [] for key in ductile_iron_brands['ASTM_A536_60-40-18'].keys()}
for brand in data['Brand']:
    composition = ductile_iron_brands[brand]
    for element, value in composition.items():
        composition_data[element].append(value)

# Додамо дані про склад у загальний словник
data.update(composition_data)

# Створимо DataFrame
df = pd.DataFrame(data)

# Збережемо дані у CSV файл
df.to_csv('synthetic_ductile_iron_data.csv', index=False)

print(df.head())  # Display the first few rows to check
