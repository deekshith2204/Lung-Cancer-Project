import json
import pickle

__data_columns=None
__model=None

def predict_lungcancer(y):

def load_server_artificts():
    print("loading artificts files ......")
    global __data_columns
    global __model

    with open("./artificts/lung_cancer_columns.json",'r') as f:
        __data_columns = json.load(f)['data_columns']

    with open("./artificts/lungcancerprediction.pickle",'rb') as f:
        __model = pickle.load(f)
    print('model id saved .. done')


if __name__ == "__main__":
    load_server_artificts()
