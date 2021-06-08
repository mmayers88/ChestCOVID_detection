import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

# https://www.kaggle.com/vijendersingh412/imagenette-classification-using-pure-tensorflow
# https://www.tensorflow.org/datasets/catalog/imagenette


def getData():
    data, info = tfds.load("imagenette/full-size-v2", with_info=True, as_supervised=True)
    train_data, valid_data = data['train'], data['validation']

    del data
    print(type(train_data))
    train_dataset = train_data.map(
        lambda image, label: (tf.image.resize(image, (224, 224)), label))

    validation_dataset = valid_data.map(
        lambda image, label: (tf.image.resize(image, (224, 224)), label)
    )
    print(type(train_dataset))
    del train_data
    del valid_data

    num_classes = info.features['label'].num_classes
    print(f'Total number of classes in dataset is {num_classes}')

    get_label_name = info.features['label'].int2str
    text_labels = [get_label_name(i) for i in range(num_classes)]
    for idx,i in enumerate(text_labels):
        print(f'The Label {idx} name is {i}')

    X_train = list(map(lambda x: x[0], train_dataset))
    y_train = list(map(lambda x: x[1], train_dataset))


    X_valid = list(map(lambda x: x[0], validation_dataset))
    y_valid = list(map(lambda x: x[1], validation_dataset))

    y_train = tf.keras.utils.to_categorical(y_train, num_classes)
    y_valid = tf.keras.utils.to_categorical(y_valid, num_classes)

    del train_dataset
    del validation_dataset

    train_len = info.splits['train'].num_examples
    valid_len = info.splits['validation'].num_examples
    print(f'Train size {train_len} and Valid size {valid_len}')

    print(type(X_train), type(y_train), type(X_valid), type(y_valid))
    return np.array(X_train), y_train, np.array(X_valid), y_valid

if __name__ == "__main__":
    train_images, train_labels, test_images, test_labels = getData()
    print(train_images.shape, train_labels.shape, test_images.shape, test_labels.shape)
    np.save('train_images.npy', train_images)
    np.save('train_labels.npy', train_labels)
    np.save('test_images.npy', test_images)
    np.save('test_labels.npy', test_labels)