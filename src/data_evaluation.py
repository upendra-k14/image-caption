import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import h5py

class DataEvaluator(object):

    def __init__(self, model,
            data_path='preprocessed_data/',
            images_path='iaprtc12/',
            test_data_filename='test_data.txt',
            word_to_id_filename='word_to_id.p',
            id_to_word_filename='id_to_word.p',
            image_name_to_features_filename='inception_image_name_to_features.h5',
            data_handler):
        self.model = model
        self.data_path = data_path
        self.images_path = images_path
        self.BOS = data_handler.BOS
        self.EOS = data_handler.EOS
        self.IMG_FEATS = data_handler.N_FEATURE
        self.MAX_TOKEN_LENGTH = data_handler.max_captions_len + 2
        self.test_data = pd.read_table(data_path +
                                       test_data_filename, sep='*')
        self.word_to_id = pickle.load(open(data_path +
                                           word_to_id_filename, 'rb'))
        self.id_to_word = pickle.load(open(data_path +
                                           id_to_word_filename, 'rb'))
        self.VOCABULARY_SIZE = len(self.word_to_id)
        self.image_names_to_features = h5py.File(data_path +
                                        image_name_to_features_filename)


    def display_caption(self, image_file=None, data_name=None):

        test_data = self.test_data

        if image_file == None:
            image_name = np.asarray(test_data.sample(1))[0][0]
        else:
            image_name = image_file
        features = self.image_names_to_features[image_name]['image_features'][:]
        text = np.zeros((1, self.MAX_TOKEN_LENGTH, self.VOCABULARY_SIZE))
        begin_token_id = self.word_to_id[self.BOS]
        text[0, 0, begin_token_id] = 1
        image_features = np.zeros((1, self.MAX_TOKEN_LENGTH, self.IMG_FEATS))
        image_features[0, 0, :] = features
        print(self.BOS)
        for word_arg in range(self.MAX_TOKEN_LENGTH - 1):
            predictions = self.model.predict([text, image_features])
            word_id = np.argmax(predictions[0, word_arg, :])
            next_word_arg = word_arg + 1
            text[0, next_word_arg, word_id] = 1
            word = self.id_to_word[word_id]
            print(word)
            if word == self.EOS:
                break
            #images_path = '../dataset/images/'
        plt.imshow(plt.imread(self.images_path + image_name))
        plt.show()

    def write_captions(self, dump_filename=None):
        if dump_filename == None:
            dump_filename = self.data_path + 'predicted_captions.txt'

        predicted_captions = open(dump_filename, 'w')

        image_names = self.test_data['image_names'].tolist()
        for image_name in image_names:

            features = self.image_names_to_features[image_name]\
                                            ['image_features'][:]
            text = np.zeros((1, self.MAX_TOKEN_LENGTH, self.VOCABULARY_SIZE))
            begin_token_id = self.word_to_id[self.BOS]
            text[0, 0, begin_token_id] = 1
            image_features = np.zeros((1, self.MAX_TOKEN_LENGTH,
                                                self.IMG_FEATS))
            image_features[0, 0, :] = features
            neural_caption = []
            for word_arg in range(self.MAX_TOKEN_LENGTH-1):
                predictions = self.model.predict([text, image_features])
                word_id = np.argmax(predictions[0, word_arg, :])
                next_word_arg = word_arg + 1
                text[0, next_word_arg, word_id] = 1
                word = self.id_to_word[word_id]
                if word == '<E>':
                    break
                else:
                    neural_caption.append(word)
            neural_caption = ' '.join(neural_caption)
            predicted_captions.write(neural_caption+'\n')
        predicted_captions.close()
        target_captions = self.test_data['caption']
        target_captions.to_csv(self.data_path + 'target_captions.txt',
                               header=False, index=False)

if __name__ == '__main__':
    from keras.models import load_model

    root_path = '../datasets/IAPR_2012/'
    data_path = root_path + 'preprocessed_data/'
    images_path = root_path + 'iaprtc12/'
    model_filename = '../trained_models/IAPR_2012/iapr_weights.90-1.99.hdf5'
    model = load_model(model_filename)
    evaluator = Evaluator(model, data_path, images_path)
    evaluator.display_caption()
