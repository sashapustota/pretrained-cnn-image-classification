import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-trs', '--train_samples', type=int, default=16000)
    parser.add_argument('-vs', '--val_samples', type=int, default=4000)
    parser.add_argument('-tes', '--test_samples', type=int, default=1000)
    parser.add_argument('-e', '--epochs', type=int, default=15)
    return parser.parse_args()

def make_dataframe_from_json(json_path):
    # Load JSON data into a list of dictionaries
    data = []
    with open(json_path) as f:
        for line in f:
            data.append(json.loads(line))

    # Convert list of dictionaries to a dataframe
    return pd.DataFrame(data)

def load_data(batch_size, train_samples, val_samples, test_samples):

    train_json_file = os.getcwd() + '/data/train_data.json'
    val_json_file = os.getcwd() + '/data/val_data.json'
    test_json_file = os.getcwd() + '/data/test_data.json'

    train_batch_size = batch_size
    val_batch_size = batch_size
    target_size = (224, 224)  # size of input images

    num_train_samples = train_samples
    num_val_samples = val_samples
    num_test_samples = test_samples

    data_folder = os.getcwd() + "/data"

    train_df = make_dataframe_from_json(train_json_file)
    # Change image paths to absolute paths
    train_df['image_path'] = train_df['image_path'].apply(lambda x: os.path.join(data_folder, x))

    val_df = make_dataframe_from_json(val_json_file)
    # Change image paths to absolute paths
    val_df['image_path'] = val_df['image_path'].apply(lambda x: os.path.join(data_folder, x))

    test_df = make_dataframe_from_json(test_json_file)
    # Change image paths to absolute paths
    test_df['image_path'] = test_df['image_path'].apply(lambda x: os.path.join(data_folder, x))

    num_classes = train_df['class_label'].nunique()

    train_df = train_df.sample(n=num_train_samples, random_state=420)
    val_df = val_df.sample(n=num_val_samples, random_state=420)
    test_df = test_df.sample(n=num_test_samples, random_state=420)

    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,  # normalize pixel values to [0,1]
    horizontal_flip=True  # apply horizontal flipping augmentation
    )

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='image_path',
        y_col='class_label',
        target_size=target_size,
        batch_size=train_batch_size,
        class_mode='categorical',
        shuffle=True,  # shuffle the data
        seed=42  # set random seed for reproducibility
    )

    val_generator = train_datagen.flow_from_dataframe(
        val_df,
        x_col='image_path',
        y_col='class_label',
        target_size=target_size,
        batch_size=val_batch_size,
        class_mode='categorical',
        shuffle=False  # don't shuffle the data
    )

    test_generator = train_datagen.flow_from_dataframe(
        test_df,
        x_col='image_path',
        y_col='class_label',
        target_size=target_size,
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator, test_generator, num_classes

def load_model(num_classes):
    # Loading InceptionV3 model
    pretrained_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    # Freezing pre-trained layers
    for layer in pretrained_model.layers:
        layer.trainable = False

    # New layers for classification
    num_classes = num_classes
    x = pretrained_model.output
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=pretrained_model.input, outputs=predictions)

    # Compile the model with an appropriate loss function, optimizer, and evaluation metric.
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model

def fit_model(batch_size, train_samples, val_samples, epochs, model, train_generator, val_generator):
    steps_per_epoch = train_samples // batch_size
    validation_steps = val_samples // batch_size
    epochs = epochs
    history = model.fit_generator(train_generator, steps_per_epoch=steps_per_epoch, epochs=epochs, validation_data=val_generator, validation_steps=validation_steps)
    return history

def plot_history(history):
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('accuracy.png')

    # Clear the plot
    plt.clf()
    
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig('loss.png')

def accuracy_report(test_generator, model):
    # Use the model to predict class probabilities for the test set
    preds = model.predict_generator(test_generator, steps=len(test_generator))
    pred = np.argmax(preds, axis=1)
    class_names = list(test_generator.class_indices.keys())
    # Save the classification report
    report = classification_report(test_generator.classes, pred, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv('classification_report.csv', index=True)

def main():
    args = parse_args()
    batch_size = args.batch_size
    train_samples = args.train_samples
    val_samples = args.val_samples
    test_samples = args.test_samples
    epochs = args.epochs

    train_generator, val_generator, test_generator, num_classes = load_data(batch_size, train_samples, val_samples, test_samples)
    model = load_model(num_classes)
    history = fit_model(batch_size, train_samples, val_samples, epochs, model, train_generator, val_generator)
    plot_history(history)
    accuracy_report(test_generator, model)

if __name__ == '__main__':
    main()

