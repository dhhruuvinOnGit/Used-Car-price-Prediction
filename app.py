from flask import Flask, render_template, request, jsonify
import pandas as pd
import os
import joblib, locale

app = Flask(__name__)

csv_path = './cleaned_car.csv'
model_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'LinearRegressionModel.pkl')
print(f"Current working directory: {os.getcwd()}")
print(f"Model path: {model_path}")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
else:
    print(f"Error: File '{csv_path}' not found.")
    df = pd.DataFrame()

if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    print(f"Error: Model file '{model_path}' not found.")
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(df['company'].unique())
    car_model = sorted(df['name'].unique())
    year = sorted(df['year'].unique(), reverse=True)
    fuel_type = df['fuel_type'].unique()

    return render_template('index.html', companies=companies, car_model=car_model, years=year, fuel_type=fuel_type)

locale.setlocale(locale.LC_NUMERIC, 'en_IN.UTF-8')

@app.route('/predict_price', methods=['POST'])
def predict_price():
    selected_company = request.form.get('company')
    selected_model = request.form.get('car_model')
    selected_year = int(request.form.get('year'))
    selected_fuel_type = request.form.get('fuel_type')
    kms_driven = float(request.form.get('kms'))

    # Create a DataFrame with the user's input
    user_input = pd.DataFrame({
        'company': [selected_company],
        'name': [selected_model],
        'year': [selected_year],
        'fuel_type': [selected_fuel_type],
        'kms_driven': [kms_driven]
    })

    # Make predictions using the model
    predicted_price = None
    if model is not None:
        predicted_price = model.predict(user_input)[0]
    
    rounded_price = round(predicted_price, 2)
    formatted_price = locale.format_string("%.2f", rounded_price, grouping=True)

    # message = f"Company: {selected_company}, Model: {selected_model}, Fuel Type: {selected_fuel_type}, Predicted Price: ₹{rounded_price}"

    details = {
        'company': selected_company,
        'model': selected_model,
        'fuel_type': selected_fuel_type,
        'predicted_price': formatted_price
    }

    return render_template('index.html', details=details)

@app.route('/get_models/<company>', methods=['GET'])
def get_models(company):
    models = sorted(df[df['company'] == company]['name'].unique())
    return jsonify(models=models)

if __name__ == "__main__":
    app.run(debug=True)