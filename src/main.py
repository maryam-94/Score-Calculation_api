import json

import pandas as pd
from flask import Flask, jsonify
from flask import request

from df_for_application import get_df_for_application
from optimum_threshold import get_optimum_threshold
from prediction_model import load_prediction_model

app = Flask(__name__)

optimum_threshold = get_optimum_threshold()
prediction_model = load_prediction_model()
df_for_application = get_df_for_application()


@app.route('/')
def hello_world():
    return 'Welcome to API prediction please call POST /predict to run client credit prediction'

@app.route('/user/random', methods=['GET'])
def get_random_user_data():
    user_data = df_for_application.sample().iloc[0].to_json()
    return app.response_class(
        response=user_data,
        mimetype='application/json'
    ), 200

# 101356 100001
@app.route('/user/<sk_id>', methods=['GET'])
def get_user_data(sk_id):
    user_data = df_for_application[df_for_application['SK_ID_CURR'] == int(sk_id)].iloc[0].to_json()
    return app.response_class(
        response=user_data,
        mimetype='application/json'
    ), 200


@app.route('/optimum_threshold', methods=['GET'])
def get_optimum():
    return jsonify({"optimum_threshold": optimum_threshold}), 200


@app.route('/predict', methods=['POST'])
def predict():
    df_one_client = pd.Series(request.json).to_frame().transpose()
    try:
        prediction_result = prediction_model.predict_proba(df_one_client)
        return jsonify(
            {
                "optimum_threshold": optimum_threshold,
                "prediction": {
                    "no_default_credit_proba": prediction_result[0, 0],
                    "default_credit_proba": prediction_result[0, 1]
                }
            }
        ), 200
    except Exception as e:
        return jsonify(e.args), 400

@app.route('/predict/user/<sk_id>', methods=['GET'])
def predict_by_sk_id(sk_id):
    df_one_client = df_for_application[df_for_application['SK_ID_CURR'] == int(sk_id)]
    if len(df_one_client) == 0:
        return jsonify({"message": 'SK_ID_CURR = ' + sk_id + ' does not exist!' }), 400
    try:
        prediction_result = prediction_model.predict_proba(df_one_client)
        return jsonify(
            {
                "optimum_threshold": optimum_threshold,
                "prediction": {
                    "no_default_credit_proba": prediction_result[0, 0],
                    "default_credit_proba": prediction_result[0, 1]
                }
            }
        ), 200
    except Exception as e:
        return jsonify(e.args), 400

if __name__ == '__main__':
    app.run()
