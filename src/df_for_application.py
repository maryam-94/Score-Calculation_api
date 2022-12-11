import pickle

def get_df_for_application():
    df_for_application = r'../data/test_df_for_application.pickle'
    f = open(df_for_application, 'rb')
    df_for_application = pickle.load(f)
    f.close()
    return df_for_application