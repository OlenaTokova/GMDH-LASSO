from flask import Flask, request, render_template, redirect, url_for, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
import joblib
from scipy.optimize import minimize
import os

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists('models'):
    os.makedirs('models')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_cooling_rate', methods=['GET', 'POST'])
def upload_cooling_rate():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            data = pd.read_csv(file_path)
            session['data'] = data.to_dict()
            return redirect(url_for('train_model'))
    return render_template('upload_cooling_rate.html')

@app.route('/train', methods=['GET'])
def train_model():
    data_dict = session.get('data')
    if data_dict:
        data = pd.DataFrame.from_dict(data_dict)
        X = data.drop(['Cooling Rate (°C/min)', 'Tensile Strength (MPa)', 'Hardness (HB)', 'Elongation (%)'], axis=1)
        y_cooling_rate = data['Cooling Rate (°C/min)']
        y_tensile_strength = data['Tensile Strength (MPa)']
        y_hardness = data['Hardness (HB)']
        y_elongation = data['Elongation (%)']
        X_train, X_test, y_train_cooling_rate, y_test_cooling_rate = train_test_split(X, y_cooling_rate, test_size=0.2, random_state=42)
        
        # Define and train GMDH model
        def gmdh(X_train, y_train, X_test):
            model = LinearRegression().fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test_cooling_rate, predictions)
            return model, predictions, mse

        # Define and train Hybrid GMDH model
        def hybrid_gmdh(X_train, y_train, X_test):
            population_size = 10
            num_generations = 5
            mutation_rate = 0.1
            best_mse = float('inf')
            best_model = None
            best_prediction = None

            for generation in range(num_generations):
                models = []
                predictions = []
                for _ in range(population_size):
                    model = Ridge(alpha=np.random.uniform(0.1, 10))
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    mse = mean_squared_error(y_test_cooling_rate, y_pred)
                    models.append((model, mse, y_pred))

                models.sort(key=lambda x: x[1])
                if models[0][1] < best_mse:
                    best_mse = models[0][1]
                    best_model = models[0][0]
                    best_prediction = models[0][2]

                for i in range(int(population_size * mutation_rate)):
                    models[-(i + 1)] = (Ridge(alpha=np.random.uniform(0.1, 10)), float('inf'), None)

            return best_model, best_prediction, best_mse

        # Train LASSO model
        lasso = Lasso(alpha=0.1)
        lasso.fit(X_train, y_train_cooling_rate)
        y_pred_lasso = lasso.predict(X_test)
        lasso_mse = mean_squared_error(y_test_cooling_rate, y_pred_lasso)

        # Get GMDH and Hybrid GMDH results
        best_model_gmdh, y_pred_gmdh, gmdh_mse = gmdh(X_train, y_train_cooling_rate, X_test)
        best_model_hybrid_gmdh, y_pred_hybrid_gmdh, hybrid_gmdh_mse = hybrid_gmdh(X_train, y_train_cooling_rate, X_test)

        mse_values = {'GMDH': gmdh_mse, 'Hybrid GMDH': hybrid_gmdh_mse, 'LASSO': lasso_mse}
        best_model_name = min(mse_values, key=mse_values.get)
        best_mse = mse_values[best_model_name]

        best_model = {
            'GMDH': best_model_gmdh,
            'Hybrid GMDH': best_model_hybrid_gmdh,
            'LASSO': lasso
        }[best_model_name]

        # Save the best model
        joblib.dump(best_model, 'models/best_model.pkl')
        session['best_model_name'] = best_model_name
        session['mse'] = best_mse

        return render_template('train_model.html', mse_values=mse_values, best_model_name=best_model_name, best_mse=best_mse)
    return redirect(url_for('upload_cooling_rate'))

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        form_data = request.form
        new_data = pd.DataFrame({
            'Austempering Temperature (°C)': [float(form_data['austempering_temperature'])],
            'Austempering Time (minutes)': [float(form_data['austempering_time'])],
            'Initial Temperature (°C)': [float(form_data['initial_temperature'])],
            'Carbon (%)': [float(form_data['carbon'])],
            'Silicon (%)': [float(form_data['silicon'])],
            'Manganese (%)': [float(form_data['manganese'])],
            'Sulfur (%)': [float(form_data['sulfur'])],
            'Phosphorus (%)': [float(form_data['phosphorus'])],
            'Chromium (%)': [float(form_data['chromium'])],
            'Nickel (%)': [float(form_data['nickel'])],
            'Molybdenum (%)': [float(form_data['molybdenum'])],
            'Tensile Strength (MPa)': [float(form_data['tensile_strength'])],
            'Hardness (HB)': [float(form_data['hardness'])],
            'Elongation (%)': [float(form_data['elongation'])],
            'Thermal Conductivity (W/m·K)': [float(form_data['thermal_conductivity'])],
            'Specific Heat (J/kg·K)': [float(form_data['specific_heat'])],
            'Melting Temperature (°C)': [float(form_data['melting_temperature'])]
        })

        if os.path.exists('models/best_model.pkl'):
            best_model = joblib.load('models/best_model.pkl')
            predicted_cooling_rate = best_model.predict(new_data)[0]
            return render_template('predict.html', form_data=form_data, predicted_cooling_rate=predicted_cooling_rate)
        else:
            return render_template('predict.html', error="Model not found. Please train the model first.")

    return render_template('predict.html')

@app.route('/upload_composition', methods=['GET', 'POST'])
def upload_composition():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(file_path)
            data = pd.read_csv(file_path)
            session['data_composition'] = data.to_dict()
            return redirect(url_for('suggest_composition'))
    return render_template('upload_composition.html')

@app.route('/suggest_composition', methods=['GET', 'POST'])
def suggest_composition():
    if request.method == 'POST':
        desired_tensile_strength = float(request.form['desired_tensile_strength'])
        desired_hardness = float(request.form['desired_hardness'])
        desired_elongation = float(request.form['desired_elongation'])

        def objective_function(params):
            model = joblib.load('models/best_model.pkl')
            params_dict = {
                'Austempering Temperature (°C)': params[0],
                'Austempering Time (minutes)': params[1],
                'Initial Temperature (°C)': params[2],
                'Carbon (%)': params[3],
                'Silicon (%)': params[4],
                'Manganese (%)': params[5],
                'Sulfur (%)': params[6],
                'Phosphorus (%)': params[7],
                'Chromium (%)': params[8],
                'Nickel (%)': params[9],
                'Molybdenum (%)': params[10],
                'Tensile Strength (MPa)': desired_tensile_strength,
                'Hardness (HB)': desired_hardness,
                'Elongation (%)': desired_elongation,
                'Thermal Conductivity (W/m·K)': params[11],
                'Specific Heat (J/kg·K)': params[12],
                'Melting Temperature (°C)': params[13]
            }
            df_params = pd.DataFrame(params_dict, index=[0])
            prediction = model.predict(df_params)
            return (prediction - desired_tensile_strength) ** 2

        initial_guess = [200, 60, 800, 3.0, 2.0, 0.3, 0.02, 0.02, 0.1, 0.1, 0.1, 30, 500, 1200]
        result = minimize(objective_function, initial_guess, method='BFGS')
        optimized_params = result.x

        suggested_composition = {
            'Austempering Temperature (°C)': optimized_params[0],
            'Austempering Time (minutes)': optimized_params[1],
            'Initial Temperature (°C)': optimized_params[2],
            'Carbon (%)': optimized_params[3],
            'Silicon (%)': optimized_params[4],
            'Manganese (%)': optimized_params[5],
            'Sulfur (%)': optimized_params[6],
            'Phosphorus (%)': optimized_params[7],
            'Chromium (%)': optimized_params[8],
            'Nickel (%)': optimized_params[9],
            'Molybdenum (%)': optimized_params[10],
            'Thermal Conductivity (W/m·K)': optimized_params[11],
            'Specific Heat (J/kg·K)': optimized_params[12],
            'Melting Temperature (°C)': optimized_params[13]
        }

        return render_template('suggest_composition.html', suggested_composition=suggested_composition)

    return render_template('suggest_composition.html')

if __name__ == '__main__':
    app.run(debug=True)

