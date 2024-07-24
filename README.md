# GMDH-LASSO
This Flask application provides a complete pipeline for uploading data, training machine learning models, making predictions, and optimizing material compositions. It uses a combination of web technologies and machine learning techniques to offer an interactive and functional tool for material science applications.
Overview
This Flask application is designed to handle file uploads, train machine learning models, make predictions, and suggest optimized compositions for given material properties. The primary focus is on predicting the cooling rate of materials based on various input parameters.

Key Components
Dependencies and Imports:

Flask: For creating web routes, handling requests, and rendering templates.
pandas, numpy: For data manipulation and numerical computations.
sklearn: For machine learning models and evaluation metrics.
joblib: For saving and loading trained models.
scipy.optimize: For optimization tasks.
os: For file system operations.
Application Configuration:

UPLOAD_FOLDER: Directory for storing uploaded files.
MODEL_FOLDER: Directory for storing trained models.
Directories are created if they do not exist.
Routes
Home Route:

@app.route('/'): Renders the home page (index.html).
Upload Cooling Rate Route:

@app.route('/upload_cooling_rate', methods=['GET', 'POST']): Handles file uploads for cooling rate data.
If a file is uploaded, it is saved to UPLOAD_FOLDER, and the data is read into a pandas DataFrame.
Data is stored in the session for later use in training.
Redirects to the model training route.
Train Model Route:

@app.route('/train', methods=['GET', 'POST']): Trains multiple machine learning models to predict cooling rate.
Retrieves data from the session and checks for required columns.
Splits data into training and testing sets.
Defines two functions for training models:
gmdh: Uses Linear Regression.
hybrid_gmdh: Uses a hybrid approach with Ridge Regression and genetic algorithms.
Trains Lasso Regression as well.
Selects the best model based on Mean Squared Error (MSE).
Saves the best model using joblib and stores information in the session.
Renders the training results (train_model.html).
Download Model Route:

@app.route('/download_model', methods=['GET']): Allows downloading the trained model.
Sends the saved model file for download.
Predict Route:

@app.route('/predict', methods=['GET', 'POST']): Makes predictions for new data.
If a POST request is received, it reads form data and creates a new DataFrame.
Loads the best model and makes predictions.
Displays the predicted cooling rate (predict.html).
Upload Composition Route:

@app.route('/upload_composition', methods=['GET', 'POST']): Handles file uploads for composition data.
Similar to the cooling rate upload route, it saves and reads uploaded data.
Suggest Composition Route:

@app.route('/suggest_composition', methods=['GET', 'POST']): Suggests optimized material compositions to achieve desired properties.
Uses an optimization function to minimize the difference between predicted and desired tensile strength.
Returns the suggested composition parameters (suggest_composition.html).
Explanation of Model Training
Data Preparation:

Data is split into features (X) and target (y_cooling_rate).
Training and testing sets are created.
Model Training:

Three models are trained: Linear Regression (GMDH), Hybrid GMDH, and Lasso Regression.
The best model is selected based on the lowest MSE.
Explanation of Optimization for Composition
Objective Function:
Uses the trained model to predict tensile strength based on composition parameters.
Optimization aims to minimize the squared difference between predicted and desired tensile strength.
