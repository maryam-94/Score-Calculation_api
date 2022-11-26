
import numpy as np
import pickle
import streamlit as st
from sklearn.preprocessing import MinMaxScaler

from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# model_path = r'ada_fbeta_joblib.pickle'
#
# model = joblib.load(model_path)

model_path = r'ada_for_predection.pickle'
f = open(model_path, 'rb')
model = pickle.load(f)
f.close()


path = r'feature_importance_ada_for_app.pickle'
f = open(path, 'rb')
all_features = pickle.load(f)
f.close()

def run():
    st.title("Credit predection Api ")
    html_temp="""
    """
    st.markdown(html_temp)

if __name__=='__main__':
    run()

# st.subheader('if you are a man tap 0 in <Gender>, if you are a woman tap 1')
st.write('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css"> ',
         unsafe_allow_html=True)

my_dict=({0:'crédit accordé ',
          1:'crédit refusé'})

def get_key(my_dict,K):
    for key,value in my_dict.items():
        if K == key:
            return value
    return "key doesn't exist " + str(value)

scaler_transform = MinMaxScaler()
def prediction_model(CODE_GENDER, DAYS_EMPLOYED, AMT_ANNUITY, DAYS_BIRTH):
    pred_args = [CODE_GENDER, DAYS_EMPLOYED, AMT_ANNUITY, DAYS_BIRTH]
    pred_arr = np.array(pred_args)
    preds = pred_arr.reshape(1, -1)
    preds = scaler_transform.fit_transform(preds)
    #   preds=preds.astype(int)
    model_prediction = model.predict(preds)
    print(model_prediction[0])
    return get_key(my_dict, model_prediction[0])



value_gender = st.selectbox('GENDER',('male','female'))
CODE_GENDER = '1' if  value_gender == 'female' else '0'

# CODE_GENDER=st.text_input('GENDER')
# print(type(CODE_GENDER))
DAYS_EMPLOYED=st.text_input('DAYS_EMPLOYED')
AMT_ANNUITY=st.text_input('CREDIT_AMOUNT')
DAYS_BIRTH=st.text_input('DAYS_BIRTH')

prediction=" "
if st.button("Predict"):
    prediction=prediction_model(CODE_GENDER, DAYS_EMPLOYED, AMT_ANNUITY, DAYS_BIRTH)
st.success("Résultat: {}".format(prediction))


