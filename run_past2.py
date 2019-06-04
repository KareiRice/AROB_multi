import numpy as np
import json
import pickle
import Model
import argparse
import torch
from collections import defaultdict
import sys
sys.path.append('../../')
#import utils
import csv
import os


def load_dataset():
    speaker_dict = defaultdict(int)
    speaker_dict['A'] = 1
    speaker_dict['B'] = 2
    emotion_dict = {'': 0, 'Neutral': 1, 'Fear': 2, 'Sadness': 3, 'Surprise': 4, 'Anger': 5, 'Disgust': 6, 'Joy': 7}
    with open('./csv_file/data_Boy_En.pkl', 'rb') as p:
        lines = pickle.load(p)
    data = []
    kara = lines[0][12]
    past_info = np.concatenate((kara, kara))
    try:
        for num, line in enumerate(lines):
            if not line[6] == '' and not emotion_dict[line[5]] == 0:
                data.append([np.concatenate((past_info, line[12])), 1 if emotion_dict[line[5]] == 1 else 0, line[6]])
            if not line[7] == '' and not emotion_dict[line[5]] == 0:
                data.append([np.concatenate((past_info, line[13])), 1 if emotion_dict[line[5]] == 1 else 0, line[7]])
            past_info = np.concatenate((line[12], line[13]))
        print(num)
    except ImportError:
        pass
    print(len(data), np.sum([datum[1] for datum in data]))
    return data


if __name__ == '__main__':
    epochs = 300
    X_VALID = 10
    SAVE_DIR = './hist_past2/extract/X{}'.format(X_VALID)
    data = load_dataset()
    np.random.shuffle(data)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # set model's parameters and the model
    EMBED_SIZE = 300
    HIDDEN_SIZE = 128
    CATEGORIES = 2
    valid_length = int(len(data) * (1 / X_VALID))

    log_train = []
    log_test = []
    log_test_categories = []
    for time in range(X_VALID):
        model = Model.StackedGRU(1, EMBED_SIZE, EMBED_SIZE*2, HIDDEN_SIZE, CATEGORIES)
        log_train.append([])
        log_test.append([])
        log_test_categories.append([])
        X, y, sentences = np.array(np.array(data)[:, 0].tolist()), np.array(np.array(data)[:, 1].tolist()), np.array(np.array(data)[:, 2].tolist())
        X_train, y_train, X_test, y_test, train_sentences, test_sentences = X[valid_length:], y[valid_length:], X[:valid_length], y[:valid_length], sentences[:valid_length], sentences[:valid_length]
        train_data_set, test_data_set = Model.data_loader(X_train, y_train, X_test, y_test)
        batch_train = Model.build_train_process(model, train_data_set)
        test = Model.build_test_process(model, test_data_set)
        test_categories = Model.build_categories_test_process(model, test_data_set, ['A', 'B'])

        data = data[valid_length:] + data[:valid_length]
        for e in range(epochs):
            log_train[time].append(batch_train(e))
            log_test[time].append(test(time, e+1, test_sentences))
            log_test_categories[time].append(test_categories())
    #utils.check_dirs('{}/'.format(SAVE_DIR))
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    with open('{}/log_train.pkl'.format(SAVE_DIR), 'wb') as p:
        pickle.dump(log_train, p)
    with open('{}/log_test.pkl'.format(SAVE_DIR), 'wb') as p:
        pickle.dump(log_test, p)
    with open('{}/log_test_categories.pkl'.format(SAVE_DIR), 'wb') as p:
        pickle.dump(log_test_categories, p)
    print('Finished Training')
