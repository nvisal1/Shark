from flask import Flask, jsonify, request
from flask_cors import CORS, cross_origin
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_transformer
import joblib

dataset_path = './50_Startups.csv'
model_path = './model.sav'

app = Flask(__name__)
CORS(app, resources={r'/*': {'origins': '*'}})

@app.route('/')
@cross_origin()
def index():
    return jsonify({'message': 'Welcome to the Shark Prediction Service'}), 200

@app.route('/predict', methods=['POST'])
@cross_origin()
def predictSalary():
    data = request.get_json()
    if 'R&D Spend' not in data or 'Administration Spend' not in data or 'Marketing Spend' not in data or 'State' not in data:
        return jsonify({'Error': 'Bad Request'}), 400
    else:
        rd_spend = data.get('R&D Spend')
        administration_spend = data.get('Administration Spend')
        marketing_spend = data.get('Marketing Spend')
        state = data.get('State')

        loaded_model = joblib.load(model_path)
        dataset = pd.read_csv(dataset_path)
        independent_variables = dataset.iloc[:, :-1].values

        # model input is R&D Spend
        independent_variables[1,0] = rd_spend
        onehotencoder = make_column_transformer((StandardScaler(), [0, 1, 2]), (OneHotEncoder(), [3]))
        independent_variables = onehotencoder.fit_transform(independent_variables)
        y_pred = loaded_model.predict([[1, independent_variables[1,0]]])

        return jsonify({'predicted_startup_annual_profit': y_pred[0]}), 200

if __name__ == '__main__':
  app.run(debug=True, host='0.0.0.0')