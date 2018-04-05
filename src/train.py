from evaluation import Evaluator
from generator import Generator
from keras.callbacks import CSVLogger
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from models import ImageCaptionModel
from data_handler import DataHandler


################################################################################
##### GLOBAL VARIABLES & CONSTANTS #############################################
################################################################################
NUM_EPOCHS = 5000
BATCH_SIZE = 256

ROOT_PATH = '../datasets/IAPR_2012/'
CAPTIONS_FILENAME = ROOT_PATH + 'IAPR_2012_captions.txt'
IMAGE_DIR = ROOT_PATH + 'iaprtc12/'
LOG_PATH = ROOT_PATH + 'preprocessed_data/'

MAX_CAPTION_LEN = 30
WORD_FREQ_THRESH = 2
IF_EXTRACT_FEATURE = False
CNN_EXTRACTOR  = 'inception'
################################################################################
################################################################################


# Load images and extract image features
data_handler = DataHandler(captions_file=CAPTIONS_FILENAME,
                            max_captions_len=MAX_CAPTION_LEN,
                            word_freq_thresh=WORD_FREQ_THRESH,
                            image_dir=IMAGE_DIR,
                            log_path=LOG_PATH,
                            if_extract_feature=IF_EXTRACT_FEATURE)

data_handler.load_preprocess()

generator = DataGenerator(
    data_path=LOG_PATH,
    batch_size=BATCH_SIZE,
    data_handler=data_handler)

print('Train Size : {}'.format(generator.training_dataset.shape[0]))
print('Validation Size : {}'.format(generator.validation_dataset.shape[0]))

model = ImageCaptionModel(max_token_length=generator.MAX_TOKEN_LENGTH,
            vocabulary_size=generator.VOCABULARY_SIZE,
            num_image_features=generator.IMG_FEATS,
            hidden_size=128,
            embedding_size=128)

model.compile(loss='categorical_crossentropy',
              optimizer = 'adam',
              metrics=['accuracy'])

print(model.summary())
print('Number of parameters:', model.count_params())

training_history_filename = preprocessed_data_path + 'training_history.log'
csv_logger = CSVLogger(training_history_filename, append=False)
model_names = ('../trained_models/IAPR_2012/' +
               'iapr_weights.{epoch:02d}-{val_loss:.2f}.hdf5')
model_checkpoint = ModelCheckpoint(model_names,
                                   monitor='val_loss',
                                   verbose=1,
                                   save_best_only=False,
                                   save_weights_only=False)

reduce_learning_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                         patience=5, verbose=1)

callbacks = [csv_logger, model_checkpoint, reduce_learning_rate]

model.fit_generator(generator=generator.flow(mode='train'),
                    steps_per_epoch=int(num_training_samples / batch_size),
                    epochs=num_epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=generator.flow(mode='validation'),
                    validation_steps=int(num_validation_samples / batch_size))

evaluator = DataEvaluator(
    model,
    data_path=LOG_PATH,
    images_path=root_path + 'iaprtc12/',
    data_handler=data_handler)

evaluator.display_caption()
