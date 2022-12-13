import torch
import argparse
from model import Model
from run import train, test
import torch.optim as optim
from earlystop import EarlyStopping
from dataloader import getDataLoader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=-1, help='the gpu will be used, e.g "0,1,2,3"')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--max_iter', type=int, default=500, help='number of iterations')
    parser.add_argument('--lr', type=float, default= 0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--maxgradnorm', type=float, default=50.0, help='maximum gradient norm')
    parser.add_argument('--hidden_layer', type=int, default=256)
    parser.add_argument('--num_layer', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=1, help='Number of head attentions.')
    parser.add_argument('--mode', type=int, default=3, help='mode of Integration Function. '
                                                            '1:ca'
                                                            '2:mul'
                                                            '3:ca mul'
                                                            '4:rasch')
    parser.add_argument('--exercise_embed_dim', type=int, default=128, help='question embedding dimensions')
    parser.add_argument('--batch_size', type=int, default=32, help='the batch size')
    parser.add_argument('--fold', type=str, default='1', help='number of fold')
    parser.add_argument('--dataset', type=str, default='assist2017', help='dataset name')

    params = parser.parse_args()
    dataset = params.dataset

    if dataset in {"assist2009_pid"}:
        params.n_question = 110
        params.batch_size = 24
        params.seqlen = 200
        params.data_dir = 'data/' + dataset
        params.data_name = dataset
        params.n_pid = 16891

    if dataset == 'assist2009_B':
        params.hasConcept = 1
        params.max_step = 200
        params.n_knowledge_concept = 110
        params.n_exercise = 16891
        params.data_dir = './data/assist2009_B'
        params.data_name = 'assist2009_B'

    if dataset == 'assist2009':
        params.hasConcept = 1
        params.max_step = 200
        params.n_knowledge_concept = 110
        params.n_exercise = 16891
        params.data_dir = './data/assist2009'
        params.data_name = 'assist2009'

    if dataset == 'assist2017':
        params.hasConcept = 1
        params.max_step = 200
        params.n_knowledge_concept = 102
        params.n_exercise = 3162
        params.data_dir = './data/assist2017'
        params.data_name = 'assist2017'

    if dataset == 'statics':
        params.hasConcept = 1
        params.max_step = 200
        params.n_knowledge_concept = 98
        params.n_exercise = 1223
        params.data_dir = './data/STATICS'
        params.data_name = 'STATICS'

    if dataset == 'synthetic':
        params.hasConcept = 0
        params.max_step = 50
        params.n_knowledge_concept = 0
        params.n_exercise = 50
        params.data_dir = './data/synthetic'
        params.data_name = 'synthetic'

    if dataset == 'assist2015':
        params.hasConcept = 0
        params.max_step = 200
        params.n_knowledge_concept = 0
        params.n_exercise = 101
        params.data_dir = './data/assist2015'
        params.data_name = 'assist2015'


    params.input = params.exercise_embed_dim * 2
    params.output = params.n_exercise
    params.embed_d = params.exercise_embed_dim

    train_data_path = params.data_dir + "/" + params.data_name + "_train"+ params.fold + ".csv"
    valid_data_path = params.data_dir + "/" + params.data_name + "_valid"+ params.fold + ".csv"
    test_data_path = params. data_dir + "/" + params.data_name + "_test"+ params.fold + ".csv"

    train_kc_data, train_respond_data, train_exercise_data, \
    valid_kc_data, valid_respose_data, valid_exercise_data, \
    test_kc_data, test_respose_data, test_exercise_data \
        = getDataLoader(train_data_path, valid_data_path, test_data_path, params)

    train_exercise_respond_data = train_respond_data * params.n_exercise + train_exercise_data
    valid_exercise_respose_data = valid_respose_data * params.n_exercise + valid_exercise_data
    test_exercise_respose_data = test_respose_data * params.n_exercise + test_exercise_data

    model = Model(params.input, params.hidden_layer, params.num_layer, params.output, params.embed_d, params.num_heads, params.dropout, params.mode, params)

    optimizer = optim.Adam(params=model.parameters(), lr=params.lr, betas=(0.9, 0.9), weight_decay=1e-5)

    if params.gpu >= 0:
        print('device: ' + str(params.gpu))
        torch.cuda.set_device(params.gpu)
        model.cuda()

    early_stopping = EarlyStopping(params.patience, verbose=True)
    for idx in range(params.max_iter):
        train_loss, train_accuracy, train_auc, train_RMSE = train(model, params, optimizer, train_kc_data, train_exercise_data, train_respond_data, train_exercise_respond_data, params.hasConcept)
        print('Epoch %d/%d, loss : %3.5f, auc : %3.5f, accuracy : %3.5f, RMSE : %3.5f' % (idx + 1, params.max_iter, train_loss, train_auc, train_accuracy, train_RMSE))
        with torch.no_grad():
            valid_loss, valid_accuracy, valid_auc, valid_RMSE  = test(model, params, valid_kc_data, valid_exercise_data, valid_respose_data, valid_exercise_respose_data, params.hasConcept)
            print('Epoch %d/%d, loss : %3.5f, valid auc : %3.5f, valid accuracy : %3.5f, RMSE : %3.5f' % (idx + 1, params.max_iter, valid_loss, valid_auc, valid_accuracy, valid_RMSE))

        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    model.load_state_dict(torch.load('checkpoint.pt'))
    with torch.no_grad():
            test_loss, test_accuracy, test_auc, test_RMSE = test(model, params, test_kc_data, test_exercise_data, test_respose_data, test_exercise_respose_data, params.hasConcept)
            print("test_auc: %.4f\t test_accuracy: %.4f\t, RMSE : %4f\t test_loss: %.4f" % (test_auc, test_accuracy, test_RMSE,test_loss))

if __name__ == "__main__":
    main()
