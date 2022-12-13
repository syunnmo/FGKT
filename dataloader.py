import sys
sys.path.append('/')
from readdata import DataReader

def getDataLoader(train_data_path, valid_data_path, test_data_path, params):
    if params.hasConcept == 1:
        handle = DataReader(train_data_path, valid_data_path, test_data_path, params.max_step, params.hasConcept)
        kc_data_train, respose_data_train, exercise_data_train = handle.getTrainData()
        kc_data_valid, respose_data_valid, exercise_data_valid = handle.getValidData()
        kc_data_test, respose_data_test, exercise_data_test = handle.getTestData()
    else:
        handle = DataReader(train_data_path, valid_data_path, test_data_path, params.max_step, params.hasConcept)
        respose_data_train, exercise_data_train = handle.getTrainDataNoConcept()
        respose_data_valid, exercise_data_valid = handle.getValidDataNoConcept()
        respose_data_test, exercise_data_test = handle.getTestDataNoConcept()
        kc_data_train = []
        kc_data_valid = []
        kc_data_test = []
    return kc_data_train, respose_data_train, exercise_data_train, kc_data_valid, respose_data_valid, exercise_data_valid, kc_data_test, respose_data_test, exercise_data_test