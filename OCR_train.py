import argparse
import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from sklearn.metrics import confusion_matrix, classification_report

def parse_arguments():
    # Define command-line arguments for specifying data directories, image dimensions, batch size, and epochs
    parser = argparse.ArgumentParser(description="Train and evaluate a CNN model for image classification.")
    parser.add_argument("--train_dir", required=True, help="Directory path for the training data.")
    parser.add_argument("--val_dir", required=True, help="Directory path for the validation data.")
    parser.add_argument("--test_dir", required=True, help="Directory path for the test data.")
    parser.add_argument("--img_width", type=int, default=50, help="Width of the input images.")
    parser.add_argument("--img_height", type=int, default=50, help="Height of the input images.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training.")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs.")
    return parser.parse_args()

def build_model(input_shape, nb_classes):
    # Define the CNN model architecture
    model = Sequential()
    model.add(Conv2D(64, (4, 4), padding='same', input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(tf.keras.layers.Normalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (4, 4), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(tf.keras.layers.Normalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (4, 4), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(tf.keras.layers.Normalization())
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (2, 2), padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D())
    model.add(tf.keras.layers.Normalization())
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    return model

def train_model(model, train_generator, validation_generator, nb_train_samples, nb_validation_samples):
    # Compile the model
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    # Define callbacks for model training
    checkpoint_callback = ModelCheckpoint(filepath='model_checkpoint_h5', monitor='val_accuracy', save_best_only=True)
    early_stopping_callback = EarlyStopping(patience=6, monitor='val_loss')

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=nb_train_samples // args.batch_size,
        epochs=args.epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // args.batch_size,
        callbacks=[early_stopping_callback, checkpoint_callback]
    )

    return model, history

def evaluate_model(model, test_generator, nb_test_samples):
    # Evaluate the model on the test data
    score = model.evaluate_generator(test_generator, nb_test_samples // args.batch_size)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    # Testing Code
    y_pred = model.predict(test_generator)
    y_true = test_generator.classes
    y_pred_classes = np.argmax(y_pred, axis=1)

    # Calculate and display confusion matrix, precision, recall, and F1-score
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    classification_rep = classification_report(y_true, y_pred_classes)

    print("Confusion Matrix:")
    print(conf_matrix)

    print("\nClassification Report:")
    print(classification_rep)

def main(args):
    # Define input shape based on the image_data_format
    if K.image_data_format() == 'channels_first':
        input_shape = (3, args.img_width, args.img_height)
    else:
        input_shape = (args.img_width, args.img_height, 3)

    # Define data generators
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Generate data using data generators
    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(args.img_width, args.img_height),
        batch_size=args.batch_size,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        args.val_dir,
        target_size=(args.img_width, args.img_height),
        batch_size=args.batch_size,
        class_mode='categorical')

    test_generator = test_datagen.flow_from_directory(
        args.test_dir,
        target_size=(args.img_width, args.img_height),
        batch_size=args.batch_size,
        class_mode='categorical')

    # Get the number of samples for training, validation, and testing
    nb_train_samples = len(train_generator.filenames)
    nb_validation_samples = len(validation_generator.filenames)
    nb_test_samples = len(test_generator.filenames)

    # Build the CNN model
    model = build_model(input_shape, len(train_generator.class_indices))

    # Train the model
    trained_model, history = train_model(model, train_generator, validation_generator, nb_train_samples,
                                         nb_validation_samples)

    # Evaluate the model
    evaluate_model(trained_model, test_generator, nb_test_samples)

if __name__ == "__main__":
    args = parse_arguments()

    # Run the main function
    main(args)
