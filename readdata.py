import numpy as np
import itertools
class DataReader():
    def __init__(self, train_path, valid_path, test_path, maxstep, hasConcept):
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.maxstep = maxstep
        self.hasConcept = hasConcept

    def getTrainData(self):
        print('loading train data...')
        kc_data= []
        ans_data =[]
        exercise_data= []
        batch = 0
        with open(self.train_path, 'r') as train:
            for student, exercise, kc, ans in itertools.zip_longest(*[train] * 4):
                batch = batch +1
                try:
                    exercise = [int(e) for e in exercise.strip().strip(',').split(',')]
                    kc = [int(k) for k in kc.strip().strip(',').split(',')]
                    ans = [int(a) for a in ans.strip().strip(',').split(',')]
                except:
                    print(batch)
                length = np.size(kc)
                slices = length//self.maxstep + (1 if length%self.maxstep > 0 else 0)
                for i in range(slices):
                    exercise_temp = np.zeros(shape=[self.maxstep])
                    kc_temp = np.zeros(shape=[self.maxstep])
                    ans_temp = np.zeros(shape=[self.maxstep])
                    if length > 0:
                        if length >= self.maxstep:
                            l = self.maxstep
                        else:
                            l = length
                        for j in range(l):
                            exercise_temp[j] = exercise[i * self.maxstep + j]
                            kc_temp[j] = kc[i*self.maxstep + j]
                            ans_temp[j] = ans[i*self.maxstep + j]
                        length = length - self.maxstep
                    exercise_data.append(exercise_temp.tolist())
                    kc_data.append(kc_temp.tolist())
                    ans_data.append(ans_temp.tolist())
            print('train--exercise done: ' + str(np.array(exercise_data).shape))
            print('train--knowledge concept done: ' + str(np.array(kc_data).shape))
            print('train--ans done: ' + str(np.array(ans_data).shape))
        return np.array(kc_data).astype(float), np.array(ans_data).astype(float), np.array(exercise_data).astype(float)

    def getValidData(self):
        print('loading valid data...')
        kc_data = []
        ans_data = []
        exercise_data = []
        batch = 0
        with open(self.valid_path, 'r') as valid:
            for student, exercise, kc, ans in itertools.zip_longest(*[valid] * 4):
                batch = batch + 1
                try:
                    exercise = [int(e) for e in exercise.strip().strip(',').split(',')]
                    kc = [int(k) for k in kc.strip().strip(',').split(',')]
                    ans = [int(a) for a in ans.strip().strip(',').split(',')]
                except:
                    print(batch)
                length = np.size(kc)
                slices = length // self.maxstep + (1 if length % self.maxstep > 0 else 0)
                for i in range(slices):
                    exercise_temp = np.zeros(shape=[self.maxstep])
                    kc_temp = np.zeros(shape=[self.maxstep])
                    ans_temp = np.zeros(shape=[self.maxstep])
                    if length > 0:
                        if length >= self.maxstep:
                            l = self.maxstep
                        else:
                            l = length
                        for j in range(l):
                            exercise_temp[j] = exercise[i * self.maxstep + j]
                            kc_temp[j] = kc[i * self.maxstep + j]
                            ans_temp[j] = ans[i * self.maxstep + j]
                        length = length - self.maxstep
                    exercise_data.append(exercise_temp.tolist())
                    kc_data.append(kc_temp.tolist())
                    ans_data.append(ans_temp.tolist())
            print('valid--exercise done: ' + str(np.array(exercise_data).shape))
            print('valid--knowledge concept done: ' + str(np.array(kc_data).shape))
            print('valid--ans done: ' + str(np.array(ans_data).shape))
        return np.array(kc_data).astype(float), np.array(ans_data).astype(float), np.array(exercise_data).astype(float)


    def getTestData(self):
        print('loading test data...')
        kc_data = []
        ans_data = []
        exercise_data = []
        batch = 0
        with open(self.test_path, 'r') as test:
            for student, exercise, kc, ans in itertools.zip_longest(*[test] * 4):
                batch = batch + 1
                try:
                    exercise = [int(e) for e in exercise.strip().strip(',').split(',')]
                    kc = [int(k) for k in kc.strip().strip(',').split(',')]
                    ans = [int(a) for a in ans.strip().strip(',').split(',')]
                except:
                    print(batch)
                length = np.size(kc)
                slices = length // self.maxstep + (1 if length % self.maxstep > 0 else 0)
                for i in range(slices):
                    exercise_temp = np.zeros(shape=[self.maxstep])
                    kc_temp = np.zeros(shape=[self.maxstep])
                    ans_temp = np.zeros(shape=[self.maxstep])
                    if length > 0:
                        if length >= self.maxstep:
                            l = self.maxstep
                        else:
                            l = length
                        for j in range(l):
                            exercise_temp[j] = exercise[i * self.maxstep + j]
                            kc_temp[j] = kc[i * self.maxstep + j]
                            ans_temp[j] = ans[i * self.maxstep + j]
                        length = length - self.maxstep
                    exercise_data.append(exercise_temp.tolist())
                    kc_data.append(kc_temp.tolist())
                    ans_data.append(ans_temp.tolist())
            print('test--exercise done: ' + str(np.array(exercise_data).shape))
            print('test--knowledge concept done: ' + str(np.array(kc_data).shape))
            print('test--ans done: ' + str(np.array(ans_data).shape))
        return np.array(kc_data).astype(float), np.array(ans_data).astype(float), np.array(exercise_data).astype(float)

    def getTrainDataNoConcept(self):
        print('loading train data...')
        ans_data =[]
        exercise_data= []
        batch = 0
        with open(self.train_path, 'r') as train:
            for student, exercise, ans in itertools.zip_longest(*[train] * 3):
                batch = batch +1
                try:
                    exercise = [int(e) for e in exercise.strip().strip(',').split(',')]
                    ans = [int(a) for a in ans.strip().strip(',').split(',')]
                except:
                    print(batch)
                length = np.size(exercise)
                slices = length//self.maxstep + (1 if length%self.maxstep > 0 else 0)
                for i in range(slices):
                    exercise_temp = np.zeros(shape=[self.maxstep])
                    ans_temp = np.zeros(shape=[self.maxstep])
                    if length > 0:
                        if length >= self.maxstep:
                            l = self.maxstep
                        else:
                            l = length
                        for j in range(l):
                            exercise_temp[j] = exercise[i * self.maxstep + j]
                            ans_temp[j] = ans[i*self.maxstep + j]
                        length = length - self.maxstep
                    exercise_data.append(exercise_temp.tolist())
                    ans_data.append(ans_temp.tolist())
            print('train--exercise done: ' + str(np.array(exercise_data).shape))
            print('train--ans done: ' + str(np.array(ans_data).shape))
        return np.array(ans_data).astype(float), np.array(exercise_data).astype(float)

    def getValidDataNoConcept(self):
        print('loading valid data...')
        ans_data = []
        exercise_data = []
        batch = 0
        with open(self.valid_path, 'r') as valid:
            for student, exercise, ans in itertools.zip_longest(*[valid] * 3):
                batch = batch + 1
                try:
                    exercise = [int(e) for e in exercise.strip().strip(',').split(',')]
                    ans = [int(a) for a in ans.strip().strip(',').split(',')]
                except:
                    print(batch)
                length = np.size(exercise)
                slices = length // self.maxstep + (1 if length % self.maxstep > 0 else 0)
                for i in range(slices):
                    exercise_temp = np.zeros(shape=[self.maxstep])
                    ans_temp = np.zeros(shape=[self.maxstep])
                    if length > 0:
                        if length >= self.maxstep:
                            l = self.maxstep
                        else:
                            l = length
                        for j in range(l):
                            exercise_temp[j] = exercise[i * self.maxstep + j]
                            ans_temp[j] = ans[i * self.maxstep + j]
                        length = length - self.maxstep
                    exercise_data.append(exercise_temp.tolist())
                    ans_data.append(ans_temp.tolist())
            print('valid--exercise done: ' + str(np.array(exercise_data).shape))
            print('valid--ans done: ' + str(np.array(ans_data).shape))
        return np.array(ans_data).astype(float), np.array(exercise_data).astype(float)

    def getTestDataNoConcept(self):
        print('loading test data...')
        ans_data = []
        exercise_data = []
        batch = 0
        with open(self.test_path, 'r') as test:
            for student, exercise, ans in itertools.zip_longest(*[test] * 3):
                batch = batch + 1
                try:
                    exercise = [int(e) for e in exercise.strip().strip(',').split(',')]
                    ans = [int(a) for a in ans.strip().strip(',').split(',')]
                except:
                    print(batch)
                length = np.size(exercise)
                slices = length // self.maxstep + (1 if length % self.maxstep > 0 else 0)
                for i in range(slices):
                    exercise_temp = np.zeros(shape=[self.maxstep])
                    ans_temp = np.zeros(shape=[self.maxstep])
                    if length > 0:
                        if length >= self.maxstep:
                            l = self.maxstep
                        else:
                            l = length
                        for j in range(l):
                            exercise_temp[j] = exercise[i * self.maxstep + j]
                            ans_temp[j] = ans[i * self.maxstep + j]
                        length = length - self.maxstep
                    exercise_data.append(exercise_temp.tolist())
                    ans_data.append(ans_temp.tolist())
            print('test--exercise done: ' + str(np.array(exercise_data).shape))
            print('test--ans done: ' + str(np.array(ans_data).shape))
        return np.array(ans_data).astype(float), np.array(exercise_data).astype(float)

