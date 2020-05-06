import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import os


def main():
    root_path = "/home/jacob/Documents/ImageClassification/racconsVredpandas"
    train_path = "/home/jacob/Documents/ImageClassification/racconsVredpandas/train"
    validate_path = "/home/jacob/Documents/ImageClassification/racconsVredpandas/validation"

    training_dir = os.path.join(root_path, 'train')
    validation_dir = os.path.join(root_path, 'validation')

    train_raccoon_dir = os.path.join(train_path, 'raccoons')
    train_redpanda_dir = os.path.join(train_path, 'redpandas')
    validation_raccoon_dir = os.path.join(validate_path, 'raccoons')
    validation_redpanda_dir = os.path.join(validate_path, 'redpandas')

    print('Raccoon Training: ', len(os.listdir(train_raccoon_dir)))
    print('Redpanda Training: ', len(os.listdir(train_redpanda_dir)))
    print('Raccoon validation: ', len(os.listdir(validation_raccoon_dir)))
    print('Redpanda validation: ', len(os.listdir(validation_redpanda_dir)))

    total_train_imgs = len(os.listdir(train_raccoon_dir)) + len(os.listdir(train_redpanda_dir))
    total_valid_imgs = len(os.listdir(validation_raccoon_dir)) + len(os.listdir(validation_redpanda_dir))

    batch_size = 128
    epochs = 20
    HEIGHT = 150
    WIDTH = 150

    train_img_generator = ImageDataGenerator(rescale=1./255,
                                             rotation_range=90,
                                             width_shift_range=.2,
                                             height_shift_range=.2,
                                             horizontal_flip=True,
                                             zoom_range=0.5)

    validation_img_generator = ImageDataGenerator(rescale=1./255)

    generated_training = train_img_generator.flow_from_directory(batch_size=batch_size,
                                                                 directory=training_dir,
                                                                 shuffle=True,
                                                                 target_size=(HEIGHT, WIDTH),
                                                                 class_mode='binary')
    generated_validation = validation_img_generator.flow_from_directory(batch_size=batch_size,
                                                                        directory=validation_dir,
                                                                        target_size=(HEIGHT, WIDTH),
                                                                        class_mode='binary')

    model = Sequential([
        Conv2D(16, 3, padding='same', activation='relu', input_shape=(HEIGHT, WIDTH, 3)),
        MaxPooling2D(),
        Dropout(0.1),
        Conv2D(32, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Conv2D(64, 3, padding='same', activation='relu'),
        MaxPooling2D(),
        Dropout(0.1),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(1)
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(
        generated_training,
        steps_per_epoch=(total_train_imgs / batch_size),
        epochs=epochs,
        validation_data=generated_validation,
        validation_steps=(total_valid_imgs / batch_size)
    )

    model.save('RacconVRedPanda.h5')


main()