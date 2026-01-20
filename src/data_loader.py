import tensorflow as tf

IMG_SIZE = (224, 224)
BATCH_SIZE = 64



AUTOTUNE = tf.data.AUTOTUNE

def load_datasets(train_dir, val_dir, seed=123):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        label_mode='int', # Recommended for SparseCategoricalCrossentropy
        seed=seed,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    # Create the validation dataset from the validation directory
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        label_mode='int', # Ensure labels are consistent
        seed=seed,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE
    )

    class_names = train_ds.class_names

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        #tf.keras.layers.RandomRotation(0.1),
        #tf.keras.layers.RandomZoom(0.1)
    ])

    train_ds = train_ds.map(lambda x, y:(data_augmentation(x),y),
                            num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names
