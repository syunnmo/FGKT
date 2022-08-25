import sys
sys.path.append('/')
from readdata import DataReader

def getDataLoader(train_data_path, valid_data_path, test_data_path, params):
    handle = DataReader(train_data_path, valid_data_path, test_data_path, params.max_step, params.n_knowledge_concept)
    kc_data_train, respose_data_train, exercise_data_train = handle.getTrainData()
    kc_data_valid, respose_data_valid, exercise_data_valid = handle.getValidData()
    kc_data_test, respose_data_test, exercise_data_test = handle.getTestData()
    return kc_data_train, respose_data_train, exercise_data_train,kc_data_valid, respose_data_valid, exercise_data_valid, kc_data_test, respose_data_test, exercise_data_test