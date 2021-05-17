import gin
import logging
import tensorflow as tf
import typing

from input_pipeline.dataloader import DataLoader

@gin.configurable
def load(window_size, window_shift) -> typing.Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, dict]:
    logging.info('Preparing dataset...')
    
    training_dataset, validation_dataset, test_dataset, dataset_info = DataLoader(window_size=window_size,
                                                                                  window_shift=window_shift).load_dataset()

    # t = list(training_dataset.as_numpy_iterator() )    
    # v = list(validation_dataset.as_numpy_iterator() )
    # d = list(test_dataset.as_numpy_iterator() )

    training_dataset, validation_dataset, test_dataset = prepare_dataset(training_dataset, validation_dataset, test_dataset)

    return training_dataset, validation_dataset, test_dataset, dataset_info


@gin.configurable
def prepare_dataset(train_dataset, validation_dataset, test_dataset,
                    prefetch, caching, batch_size, shuffle, repeat # <- configs
                    ) -> typing.Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:

    if batch_size > 0:
        train_dataset =      train_dataset.batch(batch_size)
        validation_dataset = validation_dataset.batch(batch_size)
        test_dataset =       test_dataset.batch(batch_size)

    if caching:
        train_dataset =      train_dataset.cache()
        validation_dataset = validation_dataset.cache()
        test_dataset =       test_dataset.cache()

    if shuffle:
        train_dataset =      train_dataset.shuffle(buffer_size=500, reshuffle_each_iteration=True)
        validation_dataset = validation_dataset.shuffle(buffer_size=500, reshuffle_each_iteration=True)
        test_dataset =       test_dataset.shuffle(buffer_size=500, reshuffle_each_iteration=True)

    if repeat:
        train_dataset = train_dataset.repeat(count=None) # None means indefinite repetition.

    if prefetch:
        train_dataset =      train_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        validation_dataset = validation_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        test_dataset =       test_dataset.prefetch(tf.data.experimental.AUTOTUNE)
        
    return train_dataset, validation_dataset, test_dataset
