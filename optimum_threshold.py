import pickle

def get_optimum_threshold():
    optimum_threshold_path = r'./data/seuil_optimum.joblib'
    f = open(optimum_threshold_path, 'rb')
    optimum_threshold = pickle.load(f)
    f.close()
    return optimum_threshold