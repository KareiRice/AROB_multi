import numpy as np
import json
import pickle
import Model
import argparse
import torch
import os
from collections import defaultdict
import sys
sys.path.append('../../')
import csv


def load_dataset():
    data = []

    speaker_dict = defaultdict(int)
    speaker_dict['A'] = 1
    speaker_dict['B'] = 2
    emotion_dict = {'': 0,'Neutral': 1, 'Fear': 2, 'Sadness': 3, 'Surprise': 4, 'Anger': 5, 'Disgust': 6, 'Joy': 7}
    # for keyword in ['Boy', 'Moe', 'Girl', 'Comedy', 'Youth']:
    #     with open('./csv_file/data_{}_En.pkl'.format(keyword), 'rb') as p:
    with open('./csv_file/data_Boy_En.pkl', 'rb') as p:
        lines = pickle.load(p)
    try:
        for num, line in enumerate(lines):
            if not line[6] == '' and not emotion_dict[line[5]] == 0:
                data.append([line[12], 1 if emotion_dict[line[5]] == 1 else 0, line[6]])
            if not line[7] == '' and not emotion_dict[line[5]] == 0:
                data.append([line[13], 1 if emotion_dict[line[5]] == 1 else 0, line[7]])
        print(num)
    except:
        pass
    print(len(data), np.sum([datum[1] for datum in data]))
    return data

# def load_dataset():
#     speaker_dict = defaultdict(int)
#     speaker_dict['A'] = 1
#     speaker_dict['B'] = 2
#     emotion_dict = {'': 0, 'Neutral': 1, 'Fear': 2, 'Sadness': 3, 'Surprise': 4, 'Anger': 5, 'Disgust': 6, 'Joy': 7}
#     data_pathname = "C:/Users/Jiali/PycharmProjects/AROB_En/data/"
#     filename = ['data_Boy_En', 'data_Comedy_En', 'data_Girl_En', 'data_Moe_En', 'data_Youth_En']
#     files_nb = len(filename)
#
#     for file_index in range(files_nb):
#
#         with open(data_pathname + '/pickle_files/' + filename[1] + '_Text.pkl', 'rb') as pickle_file:
#             pickle_data = pickle.load(pickle_file)
#
#         with open(data_pathname + filename[file_index] + '_Feeling.txt', 'r') as feeling_text_file:
#             feeling = [line.strip() for line in feeling_text_file]
#
#             data = []
#
#             for element in feeling_text_file:
#                 data.append([pickle_data[element], 1 if emotion_dict[feeling[element]] == 1 else 0])
#
#         print(len(data), np.sum([datum[1] for datum in data]))
#         return data


if __name__ == '__main__':
    SAVE_DIR = './hist/extract/X10'
    epochs = 300
    X_VALID = 10
    data = load_dataset()
    np.random.shuffle(data)
    # set model's parameters and the model
    EMBED_SIZE = 300
    HIDDEN_SIZE = 128
    CATEGORIES = 2
    valid_length = int(len(data) * (1 / X_VALID))

    log_train = []
    log_test = []
    log_test_categories = []
    for time in range(X_VALID):
        model = Model.NN(EMBED_SIZE, HIDDEN_SIZE, CATEGORIES)
        log_train.append([])
        log_test.append([])
        log_test_categories.append([])
        data = data[valid_length * time:] + data[:valid_length * time]
        X, y, sentences = np.array(np.array(data)[:, 0].tolist()), np.array(np.array(data)[:, 1].tolist()), np.array(np.array(data)[:, 2].tolist())
        X_train, y_train, X_test, y_test, train_sentences, test_sentences = X[valid_length:], y[valid_length:], X[:valid_length], y[:valid_length], sentences[:valid_length], sentences[:valid_length]
        train_data_set, test_data_set = Model.data_loader(X_train, y_train, X_test, y_test)
        batch_train = Model.build_train_process(model, train_data_set)
        test = Model.build_test_process(model, test_data_set)
        test_categories = Model.build_categories_test_process(model, test_data_set, ['A', 'B'])

        for e in range(epochs):
            log_train[time].append(batch_train(e))
            log_test[time].append(test(time, e+1, test_sentences))
            log_test_categories[time].append(test_categories())
    # utils.check_dirs('{}/'.format(SAVE_DIR))
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    with open('{}/log_train.pkl'.format(SAVE_DIR), 'wb') as p:
        pickle.dump(log_train, p)
    with open('{}/log_test.pkl'.format(SAVE_DIR), 'wb') as p:
        pickle.dump(log_test, p)
    with open('{}/log_test_categories.pkl'.format(SAVE_DIR), 'wb') as p:
        pickle.dump(log_test_categories, p)
    print('Finished Training')