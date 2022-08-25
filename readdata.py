import numpy as np
import itertools
class DataReader():
    def __init__(self, train_path, valid_path, test_path, maxstep, numofques):
        self.train_path = train_path
        self.valid_path = valid_path
        self.test_path = test_path
        self.maxstep = maxstep
        self.numofques = numofques

    def getTrainData(self):
        print('loading train data...')
        q_data= []
        qa_data =[]
        p_data= []
        batch = 0
        with open(self.train_path, 'r') as train:
            for student, problem, ques, ans in itertools.zip_longest(*[train] * 4):
                batch = batch +1
                try:
                    problem = [int(p) for p in problem.strip().strip(',').split(',')]
                    ques = [int(q) for q in ques.strip().strip(',').split(',')]
                    ans = [int(a) for a in ans.strip().strip(',').split(',')]
                except:
                    print(batch)
                length = np.size(ques)
                slices = length//self.maxstep + (1 if length%self.maxstep > 0 else 0)
                for i in range(slices):
                    q_temp = np.zeros(shape=[self.maxstep])
                    qa_temp = np.zeros(shape=[self.maxstep])
                    p_temp = np.zeros(shape=[self.maxstep])
                    if length > 0:
                        if length >= self.maxstep:
                            l = self.maxstep
                        else:
                            l = length
                        for j in range(l):
                            q_temp[j] = ques[i*self.maxstep + j]
                            qa_temp[j] = ans[i*self.maxstep + j]
                            p_temp[j] = problem[i*self.maxstep + j]
                        length = length - self.maxstep
                    q_data.append(q_temp.tolist())
                    qa_data.append(qa_temp.tolist())
                    p_data.append(p_temp.tolist())
            print('train--question done: ' + str(np.array(q_data).shape))
            print('train--question_ans done: ' + str(np.array(qa_data).shape))
            print('train--problem done: ' + str(np.array(p_data).shape))
            return np.array(q_data).astype(float), np.array(qa_data).astype(float), np.array(p_data).astype(float)

    def getValidData(self):
        print('loading valid data...')
        q_data= []
        qa_data =[]
        p_data= []
        batch = 0
        with open(self.valid_path, 'r') as valid:
            for student, problem, ques, ans in itertools.zip_longest(*[valid] * 4):
                batch = batch +1
                try:
                    problem = [int(p) for p in problem.strip().strip(',').split(',')]
                    ques = [int(q) for q in ques.strip().strip(',').split(',')]
                    ans = [int(a) for a in ans.strip().strip(',').split(',')]
                except:
                    print(batch)
                length = np.size(ques)
                slices = length//self.maxstep + (1 if length%self.maxstep > 0 else 0)
                for i in range(slices):
                    q_temp = np.zeros(shape=[self.maxstep])
                    qa_temp = np.zeros(shape=[self.maxstep])
                    p_temp = np.zeros(shape=[self.maxstep])
                    if length > 0:
                        if length >= self.maxstep:
                            l = self.maxstep
                        else:
                            l = length
                        for j in range(l):
                            q_temp[j] = ques[i*self.maxstep + j]
                            qa_temp[j] = ans[i*self.maxstep + j]
                            p_temp[j] = problem[i*self.maxstep + j]
                        length = length - self.maxstep
                    q_data.append(q_temp.tolist())
                    qa_data.append(qa_temp.tolist())
                    p_data.append(p_temp.tolist())
            print('valid--question done: ' + str(np.array(q_data).shape))
            print('valid--question_ans done: ' + str(np.array(qa_data).shape))
            print('valid--problem done: ' + str(np.array(p_data).shape))
            return np.array(q_data).astype(float), np.array(qa_data).astype(float), np.array(p_data).astype(float)


    def getTestData(self):
        print('loading test data...')
        q_data = []
        qa_data = []
        p_data = []
        batch = 0
        with open(self.test_path, 'r') as test:
            for student, problem, ques, ans in itertools.zip_longest(*[test] * 4):
                batch = batch + 1
                try:
                    problem = [int(p) for p in problem.strip().strip(',').split(',')]
                    ques = [int(q) for q in ques.strip().strip(',').split(',')]
                    ans = [int(a) for a in ans.strip().strip(',').split(',')]
                except:
                    print(batch)
                length = np.size(ques)
                slices = length // self.maxstep + (1 if length % self.maxstep > 0 else 0)
                for i in range(slices):
                    q_temp = np.zeros(shape=[self.maxstep])
                    qa_temp = np.zeros(shape=[self.maxstep])
                    p_temp = np.zeros(shape=[self.maxstep])
                    if length > 0:
                        if length >= self.maxstep:
                            l = self.maxstep
                        else:
                            l = length
                        for j in range(l):
                            q_temp[j] = ques[i * self.maxstep + j]
                            qa_temp[j] = ans[i * self.maxstep + j]
                            p_temp[j] = problem[i * self.maxstep + j]
                        length = length - self.maxstep
                    q_data.append(q_temp.tolist())
                    qa_data.append(qa_temp.tolist())
                    p_data.append(p_temp.tolist())
            print('test--question done: ' + str(np.array(q_data).shape))
            print('test--question_ans done: ' + str(np.array(qa_data).shape))
            print('test--problem done: ' + str(np.array(p_data).shape))
            return np.array(q_data).astype(float), np.array(qa_data).astype(float), np.array(p_data).astype(float)

