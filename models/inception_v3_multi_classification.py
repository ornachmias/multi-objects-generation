import os
import pickle

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model
import pandas as pd
import numpy as np
from PIL import Image

from utils.images_utils import ImagesUtils


class InceptionV3MultiClassification:
    def __init__(self, image_size, train_metadata_path=None, eval_metadata_path=None, learning_rate=0.0001,
                 dropout_rate=0.2, loss='categorical_crossentropy', batch_size=20, train_all=False, is_eval=False,
                 model_tag=''):
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.image_size = image_size
        self.loss = loss
        self.batch_size = batch_size
        self.model = None
        self.name = 'inception_v3_multi_classification'
        if model_tag != '':
            self.name += '_' + model_tag
        self.train_metadata_path = train_metadata_path
        self.eval_metadata_path = eval_metadata_path
        self.train_all = train_all
        self.train_generator = None
        self.eval_generator = None
        self.callback_handler = None
        self.is_eval = is_eval

    def get_data_generators(self):
        df_train = pd.read_csv(self.train_metadata_path)
        df_train['labels'] = df_train['categories'].apply(self.split_column)
        df_eval = pd.read_csv(self.eval_metadata_path)
        df_eval['labels'] = df_eval['categories'].apply(self.split_column)

        classes = ';'.join(df_train['categories']) + ';' + ';'.join(df_eval['categories'])
        classes = list(set(classes.split(';')))

        df_train = df_train[df_train.apply(self.check_path_row_exists, axis=1)]
        df_eval = df_eval[df_eval.apply(self.check_path_row_exists, axis=1)]

        train_datagen = ImageDataGenerator(rescale=1./255.,
                                           rotation_range=40,
                                           width_shift_range=0.2,
                                           height_shift_range=0.2,
                                           shear_range=0.2,
                                           zoom_range=0.2,
                                           horizontal_flip=True)
        eval_datagen = ImageDataGenerator(rescale=1.0/255.)

        train_generator = train_datagen.flow_from_dataframe(df_train,
                                                            x_col='path',
                                                            y_col='labels',
                                                            batch_size=self.batch_size,
                                                            class_mode='categorical',
                                                            target_size=(self.image_size, self.image_size),
                                                            classes=classes)
        validation_generator = eval_datagen.flow_from_dataframe(df_eval,
                                                                x_col='path',
                                                                y_col='labels',
                                                                batch_size=self.batch_size,
                                                                class_mode='categorical',
                                                                target_size=(self.image_size, self.image_size),
                                                                shuffle=False,
                                                                classes=classes)
        return train_generator, validation_generator

    def check_path_row_exists(self, row):
        return os.path.exists(row['path'])

    def split_column(self, categories_str):
        return categories_str.split(';')

    def init(self, callback_handler):
        self.callback_handler = callback_handler
        pre_trained_model = InceptionV3(input_shape=(self.image_size, self.image_size, 3),
                                        include_top=False, weights='imagenet')

        if not self.is_eval:
            self.train_generator, self.eval_generator = self.get_data_generators()
            self.callback_handler.set_trained_classes(self.train_generator.class_indices)

        number_of_classes = len(self.callback_handler.get_trained_classes())

        if not self.train_all:
            for layer in pre_trained_model.layers:
                layer.trainable = False

        x = layers.Flatten()(pre_trained_model.output)
        x = layers.Dense(1024, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(number_of_classes, activation='sigmoid')(x)
        self.model = Model(pre_trained_model.input, x)
        self.model.compile(optimizer=RMSprop(lr=self.learning_rate), loss=self.loss, metrics=['acc'])
        print(self.model.summary())

    def train(self, epochs=100):
        self.load_model()

        history = self.model.fit(x=self.train_generator, validation_data=self.eval_generator,
                                 batch_size=self.batch_size, epochs=epochs, verbose=1,
                                 callbacks=self.callback_handler.get_callbacks(),
                                 steps_per_epoch=len(self.train_generator),
                                 validation_steps=len(self.eval_generator))
        return history

    def eval(self, image_path):
        self.load_model()

        image = Image.open(image_path)
        image = image.resize((self.image_size, self.image_size))
        image = ImagesUtils.convert_to_numpy(image)
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))[:, :, :, :3]
        pred = self.model.predict(image)[0]
        predicted_class_indices = np.argsort(-pred)[:5]
        labels = dict((v, k) for k, v in self.callback_handler.get_trained_classes().items())
        predictions = [labels[k] for k in predicted_class_indices]
        return predictions

    def load_model(self):
        try:
            print('Found checkpoints in {}, loading...'.format(self.callback_handler.checkpoint_path))
            self.model.load_weights(self.callback_handler.checkpoint_path)
        except Exception as e:
            print('Failed to load checkpoints!')
            print(e)

    def save_model(self):
        try:
            self.model.save_weights(self.callback_handler.checkpoint_path)
        except Exception as e:
            print('Failed to load checkpoints!')
            print(e)

