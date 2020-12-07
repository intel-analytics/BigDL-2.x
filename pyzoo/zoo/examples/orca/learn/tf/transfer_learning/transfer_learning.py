import os
import tensorflow as tf
from tensorflow import keras as K

from zoo.orca import init_orca_context, stop_orca_context
from zoo.orca.learn.tf.estimator import Estimator


def train_val(cluster_mode, epochs, batch_size):
    dataset_dir = "./"
    tf.keras.utils.get_file(
        origin="https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip",
        fname="cats_and_dogs_filtered.zip", extract=True, cache_dir=dataset_dir)
    base_dir = "datasets/cats_and_dogs_filtered"

    if cluster_mode == "local":
        init_orca_context(cluster_mode="local", cores=4, memory="3g")
    elif cluster_mode == "yarn":
        additional ="datasets/cats_and_dogs_filtered.zip#datasets"
        init_orca_context(cluster_mode="yarn-client", num_nodes=2, cores=2, driver_memory="3g",
                          additional_archive=additional)

    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')

    # Directory with our training cat pictures
    train_cats_dir = os.path.join(train_dir, 'cats')
    print('Total training cat images:', len(os.listdir(train_cats_dir)))

    # Directory with our training dog pictures
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    print('Total training dog images:', len(os.listdir(train_dogs_dir)))

    # Directory with our validation cat pictures
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    print('Total validation cat images:', len(os.listdir(validation_cats_dir)))

    # Directory with our validation dog pictures
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    print('Total validation dog images:', len(os.listdir(validation_dogs_dir)))

    image_size = 160  # All images will be resized to 160x160

    IMG_SHAPE = (image_size, image_size, 3)

    def _parse_function(filename, label):
        image_string = tf.io.read_file(filename)
        image_decoded = tf.io.decode_jpeg(image_string)
        image_resized = tf.image.resize(image_decoded, [image_size, image_size])
        return tf.cast(tf.reshape(image_resized, (image_size, image_size, 3)), dtype=tf.float32) / 255.0, label

    # datasets
    train_list = [os.path.join(train_cats_dir, name) for name in os.listdir(train_cats_dir)] + [
        os.path.join(train_dogs_dir, name) for name in os.listdir(train_dogs_dir)]
    train_filenames = tf.constant(train_list)
    train_labels = tf.constant(
        [0 for x in range(len(os.listdir(train_cats_dir)))] + [1 for x in range(len(os.listdir(train_dogs_dir)))])
    train_dataset = tf.data.Dataset.from_tensor_slices((train_filenames, train_labels))
    train_dataset = train_dataset.map(_parse_function).shuffle(buffer_size=1000)

    validation_list = [os.path.join(validation_cats_dir, name) for name in os.listdir(validation_cats_dir)] + [
        os.path.join(validation_dogs_dir, name) for name in os.listdir(validation_dogs_dir)]
    validation_filenames = tf.constant(validation_list)
    validation_labels = tf.constant(
        [0 for x in range(len(os.listdir(validation_cats_dir)))] + [1 for x in
                                                                    range(len(os.listdir(validation_dogs_dir)))])
    validation_dataset = tf.data.Dataset.from_tensor_slices((validation_filenames, validation_labels))
    validation_dataset = validation_dataset.map(_parse_function).shuffle(buffer_size=1000)

    # build models
    inputs = K.layers.Input(shape=IMG_SHAPE)
    mnet = K.applications.MobileNetV2(input_tensor=inputs,
                                      include_top=False,
                                      weights='imagenet')
    mnet.trainable = False
    outputs_0 = mnet(inputs)
    outputs_1 = K.layers.GlobalAveragePooling2D()(outputs_0)
    outputs_2 = K.layers.Dense(1, activation='sigmoid')(outputs_1)
    model = K.models.Model(inputs=[inputs], outputs=[outputs_2])
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.0001),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    len(model.trainable_variables)

    # train
    print("Freeze Network")
    est = Estimator.from_keras(keras_model=model)
    est.fit(train_dataset,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_dataset
            )
    result = est.evaluate(validation_dataset)
    print(result)

    mnet.trainable = True
    # Let's take a look to see how many layers are in the base model
    print("Number of layers in the base model: ", len(mnet.layers))

    # Fine tune from this layer onwards
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in mnet.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()
    print(len(model.trainable_variables))

    print("Unfreeze Network")
    est = Estimator.from_keras(keras_model=model)
    est.fit(data=train_dataset,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=validation_dataset
            )
    result = est.evaluate(validation_dataset)
    print(result)
    stop_orca_context()

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--cluster_mode', type=str, default="local",
                        help='The mode for the Spark cluster. local or yarn.')
    parser.add_argument('--epochs', type=int, default=2,
                        help="The number of epochs to train the model")
    parser.add_argument('--batch_size', type=int, default=32,
                        help="Batch size for training and prediction")
    args = parser.parse_args()
    train_val(args.cluster_mode, args.epochs, args.batch_size)
