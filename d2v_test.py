import gensim
from gensim.models.doc2vec import Doc2Vec
import pickle
import os, os.path

model = gensim.models.doc2vec.Doc2Vec.load("d2v.model")
pathname = '/data'
file_nb = len([name for name in os.listdir(pathname) if os.path.isfile(os.path.join(pathname, name))])
filename = ['data_Boy_En_Text', 'data_Comedy_En_Text', 'data_Girl_En_Text', 'data_Moe_En_Text', '_Text']

for text_index in range (file_nb - 1):
    pickle_file = open(filename[text_index] + "_pickle", "rb")
    text = open(filename[text_index] + ".txt")
    while True:
        line = text.readline()
        result = model.infer_vector(line.split(''))
        pickle.dump(result, pickle_file)
        pickle_file.close()
        if not line:
            break
    text.close()
