import tensorflow as tf
tf.get_logger().setLevel('ERROR')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
import numpy as np
import os
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Dense
from tensorflow.keras.layers import AvgPool2D, GlobalAveragePooling2D, MaxPool2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import ReLU, concatenate
import tensorflow.keras.backend as K

#https://towardsdatascience.com/creating-densenet-121-with-tensorflow-edbc08a956d8

# https://colab.research.google.com/drive/1v2p228o-_PRtecU0vYUXuGlG_VierqcP#scrollTo=HGkSJI41fHfg&forceEdit=true&sandboxMode=true
# (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
# train_ds, test_ds = tfds.load('imagenet2012_subset', split=['train', 'test'])

batch_size = 16
input_shape = 224, 224, 3
n_classes = 10
num_blocks = 3
num_layers_per_block = 4
growth_rate = 16
dropout_rate = 0.4
compress_factor = 0.5
eps = 1.1e-5
EPOCHS = 50

num_filters = 16

train_images = np.load('train_images.npy')
train_labels = np.load('train_labels.npy')
test_images = np.load('test_images.npy')
test_labels = np.load('test_labels.npy')


print(train_images.shape,test_images.shape)

train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels)).batch(batch_size=batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(batch_size=batch_size)



# Creating Densenet121
def densenet(input_shape, n_classes, filters=32):
    # batch norm + relu + conv
    def bn_rl_conv(x, filters, kernel=1, strides=1):

        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = Conv2D(filters, kernel, strides=strides, padding='same')(x)
        return x

    def dense_block(x, repetition):

        for _ in range(repetition):
            y = bn_rl_conv(x, 4 * filters)
            y = bn_rl_conv(y, filters, 3)
            x = concatenate([y, x])
        return x

    def transition_layer(x):

        x = bn_rl_conv(x, K.int_shape(x)[-1] // 2)
        x = AvgPool2D(2, strides=2, padding='same')(x)
        return x

    input = Input(input_shape)
    x = Conv2D(64, 7, strides=2, padding='same')(input)
    x = MaxPool2D(3, strides=2, padding='same')(x)

    for repetition in [6, 12, 24, 16]:
        d = dense_block(x, repetition)
        x = transition_layer(d)
    x = GlobalAveragePooling2D()(d)
    output = Dense(n_classes, activation='softmax')(x)

    model = Model(input, output)
    return model



model = densenet(input_shape, n_classes)
model.summary()
model.compile(loss=tf.keras.losses.categorical_crossentropy, optimizer=tf.keras.optimizers.Adam(lr=0.0001), metrics=['acc'])
print(model.summary())

# Comment out the below line if you want to have an image of your model's structure.

# tf.keras.utils.plot_model( model , show_shapes=True )

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "training/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)


# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path,
    verbose=1,
    save_weights_only=True,
    save_freq=10*(train_images.shape[0]//batch_size))

latest = tf.train.latest_checkpoint(checkpoint_dir)
print("LATEST: ", latest)
# Save the weights using the `checkpoint_path` format
model.save_weights(checkpoint_path.format(epoch=0))



if latest:
    print("Reloading Weights")
    model.load_weights(latest)
    loss, acc = model.evaluate(test_dataset, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    # Train the model with the new callback
    print("Now Retraining")

else:
    print("First Run!")
    print("Now Training")


model.fit(train_dataset,
          epochs=EPOCHS,
          batch_size=batch_size,
          callbacks=[cp_callback],
          validation_data=test_dataset,
          validation_freq = 10, 
          verbose=1)

