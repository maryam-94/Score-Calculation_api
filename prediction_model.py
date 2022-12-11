import joblib


def load_prediction_model():
    model = joblib.load('./data/clf_fit.joblib')
    return model
