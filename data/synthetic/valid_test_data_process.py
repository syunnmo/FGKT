from tqdm import tqdm
import numpy as np
import itertools
import pickle
import csv
from sklearn.model_selection import train_test_split


dataset='synthetic'
fold = '5'
address = './' +  dataset + '_valid_test' + fold + '.csv'
save_path = './' +  dataset
print('loading data...')

learners=[]
with open(address, 'r') as data:
    for length, exercise, ans in tqdm(itertools.zip_longest(*[data] * 3)):
        sequence = []
        length= length.strip()
        exercise = [int(p) for p in exercise.strip().strip(',').split(',')]
        ans = [int(a) for a in ans.strip().strip(',').split(',')]

        list_length = []
        list_length.append(length)

        sequence.append(list_length)
        sequence.append(exercise)
        sequence.append(ans)

        learners.append(sequence)

    valid_data, test_data = train_test_split(learners, test_size=0.5, random_state= (10 + int(fold)) )


with open(save_path + '_'+ 'valid'+ fold +'.csv', 'w', encoding='utf-8', newline='') as fp:
    valid_data_done = []
    for learner in valid_data:
        for item in learner:
            valid_data_done.append(item)

    writer = csv.writer(fp)
    writer.writerows(valid_data_done)
    fp.close()

with open(save_path + '_'+ 'test'+ fold +'.csv', 'w', encoding='utf-8', newline='') as fp:
    test_data_done = []
    for learner in test_data:
        for item in learner:
            test_data_done.append(item)

    writer = csv.writer(fp)
    writer.writerows(test_data_done)
    fp.close()




