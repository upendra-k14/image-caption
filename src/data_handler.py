from collections import Counter
from itertools import chain
import os
import pickle
from string import digits
import time

import h5py
import numpy as np
import pandas as pd

from keras.preprocessing import image
from keras.models import Model
from keras.applications.inception_v3 import preprocess_input
from keras.applications import InceptionV3

class DataHandler():
    """
    Preprocess data and extract features
    """

    def __init__(self, captions_file, max_captions_len=20, word_freq_thresh=2,
                 if_extract_feature=False, image_dir=None, log_path,
                 separator='*'):

        # Member variables
        self.captions_file = captions_file
        self.max_captions_len = max_captions_len
        self.sep = separator
        self.word_freq_thresh = word_freq_thresh
        self.if_extract_feature = if_extract_feature
        self.image_dir = image_dir
        self.log_path = log_path

        self.BOS = '<S>' # Start of Sentence Value
        self.EOS = '<E>' # End Of Sentence Value
        self.PAD = '<P>' # Pad value
        self.N_FEATURE = 2048

        if self.if_extract_feature == True:
            assert self.image_dir != None

    def load_preprocess(self):
        """
        Preprocessing in stages
        """

        # Load data ############################################################
        print('Loading data ...')
        data = pd.read_table(self.captions_file, sep=self.sep)
        data = np.asarray(data)
        np.random.shuffle(data)
        self.im_files = data[:, 0]
        self.captions = data[:, 1]
        print('Number of instances loaded', self.im_files.shape[0])
        ########################################################################

        # Filter captions based on caption length ##############################
        print('Filtering captions ...')
        temp_im_files = []
        temp_captions = []
        prev_size = self.im_files.shape[0]

        for i in range(self.im_files.shape[0]):
            cleaned_caption = self.clean_str(self.captions[i])
            if (len(cleaned_caption) <= self.max_caption_len):
                temp_captions.append(cleaned_caption)
                temp_im_files.append(self.im_files[i])

        self.captions = temp_captions
        self.im_files = temp_im_files
        curr_size = len(self.captions)
        print('Current number of files:', current_size)
        self.current_number_of_captions = curr_size
        ########################################################################


        # Construct vocabulary and inverse map #################################
        print("Constructing vocabulary ...")
        self.remove_rare_words()
        words = self.word_frequencies[:, 0]
        self.w2i_map = {self.PAD:0, self.BOS:1, self.EOS:2}
        self.w2i_map.update({word:word_id for word_id, word
                                in enumerate(words, 3)})
        self.i2w_map = {word_id:word for word, word_id
                                in self.w2i_map.items()}
        ########################################################################


        ###### Save data #######################################################
        original_directory = os.getcwd()
        # Change current working directory
        log_dir = self.log_path
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        os.chdir(log_dir)

        if self.if_extract_feature:
            self.get_image_features(self.image_dir)
            self.save_image_features()

        # Save image_name caption file
        with open('complete_data.txt','w') as data_file
            data_file.write('image_names*caption\n')
            for i, image_name in enumerate(self.im_files):
                data_file.write('{}*{}\n'.format(
                    image_name,
                    ' '.join(self.captions[i])))

        # Save vocabulary files
        pickle.dump(self.w2i_map, open('word_to_id.p', 'wb'))
        pickle.dump(self.i2w_map, open('id_to_word.p', 'wb'))

        self.split_data()

        # Move back to earlier path
        os.chdir(original_directory)
        ########################################################################

    def clean_str(self, caption):
        incorrect_chars = digits + ";.,'/*?¿><:{}[\]|+"
        char_translator = str.maketrans('', '', incorrect_chars)
        quotes_translator = str.maketrans('', '', '"')
        clean_caption = caption.strip().lower()
        clean_caption = clean_caption.translate(char_translator)
        clean_caption = clean_caption.translate(quotes_translator)
        clean_caption = clean_caption.split(' ')
        return clean_caption

    def remove_rare_words(self):

        print('Removing rare words ...')
        self.word_frequencies = Counter(chain(*self.captions)).most_common()
        for i, freq_data in enumerate(self.word_frequencies):
            freq = freq_data[1]
            if freq <= self.word_freq_tresh:
                index_range = i
                break

        if self.word_freq_tresh != 0:
            self.word_frequencies = np.asarray(
                        self.word_frequencies[0:index_range])
        else:
            self.word_frequencies = np.asarray(self.word_frequencies)

        vocab_size = self.word_frequencies.shape[0]
        print('Current number of words:',vocab_size)
        self.current_number_of_words = vocab_size


    def extract_im_features(self, image_dir):

        base_model = InceptionV3(weights='imagenet')
        inception_model =  Model(
            input=base_model.input,
            output=base_model.get_layer('flatten').output)
        self.extracted_features = []
        self.image_feature_files = list(set(self.im_files))
        number_of_images = len(self.image_feature_files)
        for i in range(number_of_images):
            image_path = ''.join(image_dir, self.image_feature_files[i])
            if i%500 == 0:
                print('%.2f %% completed' %
                        round(100*i/number_of_images,2))
            img = image.load_img(image_path, target_size=(299, 299))
            img = np.expand_dims(image.img_to_array(img), axis=0)
            CNN_features = model.predict(preprocess_input(img))
            self.extracted_features.append(np.squeeze(CNN_features))

        self.extracted_features = np.asarray(self.extracted_features)

    def save_image_features(self):

        print('Saving image features ...')

        dataset_file = h5py.File('inception_image_name_to_features.h5')
        for i, image_file in enumerate(self.image_feature_files):
            file_id = dataset_file.create_group(image_file)
            image_data = file_id.create_dataset(
                'image_features',
                (self.N_FEATURE,),
                dtype='float32')
            image_data[:] = self.extracted_features[i,:]

        dataset_file.close()

    def write_image_feature_files(self):
        pickle.dump(self.image_feature_files,
                    open('image_feature_files.p', 'wb'))
        pickle.dump(self.extracted_features,
                    open('extracted_features.p', 'wb'))

    def split_data(self, train_percentage=.80):

        complete_data = pd.read_table('complete_data.txt',sep='*')
        data_size = complete_data.shape[0]
        training_size = int(data_size*train_porcentage)
        complete_training_data = complete_data[0:training_size]
        test_data = complete_data[training_size:]
        test_data.to_csv('test_data.txt',sep='*',index=False)
        # splitting between validation and training
        training_size = int(training_size*train_percentage)
        validation_data = complete_training_data[training_size:]
        training_data = complete_training_data[0:training_size]
        validation_data.to_csv('validation_data.txt',sep='*',index=False)
        training_data.to_csv('training_data.txt',sep='*',index=False)

if __name__ == '__main__':

    root_path = '../datasets/IAPR_2012/'
    captions_filename = root_path + 'IAPR_2012_captions.txt'
    data_handler = DataHandler(captions_file=captions_filename,
                                max_captions_len=50,
                                word_freq_thresh=2,
                                image_dir=root_path + 'iaprtc12/',
                                log_path=root_path + 'preprocessed_data/',
                                if_extract_feature=True)
    data_handler.load_preprocess()
