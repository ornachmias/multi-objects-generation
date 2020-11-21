import os

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras import layers
from tensorflow.keras import Model
import pandas as pd


class InceptionV3MultiClassification:
    def __init__(self, train_metadata_path, eval_metadata_path, image_size, learning_rate=0.0001,
                 dropout_rate=0.2, loss='categorical_crossentropy', batch_size=20, train_all=False):
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.image_size = image_size
        self.loss = loss
        self.batch_size = batch_size
        self.model = None
        self.name = 'inception_v3_multi_classification'
        self.train_metadata_path = train_metadata_path
        self.eval_metadata_path = eval_metadata_path
        self.train_all = train_all
        self.train_generator = None
        self.eval_generator = None

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
        return train_generator, validation_generator, classes

    def check_path_row_exists(self, row):
        return os.path.exists(row['path'])

    def split_column(self, categories_str):
        return categories_str.split(';')

    def init(self):
        self.train_generator, self.eval_generator, classes = self.get_data_generators()
        number_of_classes = len(classes)
        pre_trained_model = InceptionV3(input_shape=(self.image_size, self.image_size, 3),
                                        include_top=False, weights='imagenet')

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

    def train(self, callbacks, epochs=100):
        history = self.model.fit(x=self.train_generator, validation_data=self.eval_generator,
                                 batch_size=self.batch_size, epochs=epochs, verbose=1,
                                 callbacks=callbacks.get_callbacks(),
                                 steps_per_epoch=len(self.train_generator),
                                 validation_steps=len(self.eval_generator))
        return history
