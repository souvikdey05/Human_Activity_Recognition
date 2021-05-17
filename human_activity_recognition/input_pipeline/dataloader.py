import gin
import pathlib
import logging
import tensorflow as tf
import numpy as np
import scipy.stats
import typing
import re
import zipfile
import pandas as pd


# Debug options
PRINT_DATASET_SIZE = False
PRINT_DATASET_DISTIBUTION = False


@gin.configurable
class DataLoader:

    def __init__(self,
                 dataset_name, dataset_directory, tfrecords_directory, create_tfrecords, # <- configs.                                  
                 window_size, window_shift,
                 sequence_to_label, sensor_position           # <- configs.
                 ) -> None:
        
        dataset_names = ('hapt', 'rw')
        if dataset_name in dataset_names:
            self.dataset_name = dataset_name
        else:
            raise ValueError(f"Received invalid dataset name: '{dataset_name}', valid dataset names: {dataset_names}.")

        self.dataset_directory = pathlib.Path(dataset_directory)
        if (not self.dataset_directory.exists() ) or (not self.dataset_directory.is_dir() ):
            raise ValueError(f"Received invalid dataset directory: '{self.dataset_directory}'.")

        self.tfrecords_directory = pathlib.Path(tfrecords_directory)
        if not self.tfrecords_directory.exists():
            self.tfrecords_directory.mkdir(parents=True, exist_ok=True)
        
        self.create_tfrecords =  bool(create_tfrecords)
        self.sequence_to_label = bool(sequence_to_label)

        if window_size > 0:
            self.window_size = int(window_size)
        else:
            raise ValueError(f"Received invalid window size: '{window_size}', the window size has to be >0.")

        if window_shift > 0:
            self.window_shift = window_shift
        else:
            raise ValueError(f"Received invalid window shift: '{window_shift}', the window shift has to be >0.")
       
        sensor_positions = ('head', 'upperarm', 'forearm', 'chest', 'waist', 'thigh', 'shin')
        if dataset_name == 'rw': # Sensor position is unimportant for other datasets.
            if sensor_position in sensor_positions:
                self.sensor_position =   sensor_position
            else:
                raise ValueError(f"Received invalid sensor position: '{sensor_position}', valid sensor positions: {sensor_positions}.")
                        

    def load_dataset (self) -> typing.Tuple[tf.data.TFRecordDataset, tf.data.TFRecordDataset, tf.data.TFRecordDataset, dict]:
        global PRINT_DATASET_DISTIBUTION

        if self.dataset_name == 'hapt':
            if self.create_tfrecords:
                self._create_hapt_dataset()

            training_dataset, validation_dataset, test_dataset = self._load_tfrecords()
            
        elif self.dataset_name == 'rw':
            if self.create_tfrecords:
                self._create_rw_dataset()
            
            training_dataset, validation_dataset, test_dataset = self._load_tfrecords()
        
        else:
            raise ValueError(f"Unknown dataset name: '{self.dataset_name}'.")


        dataset_info = {
            'dataset_name': self.dataset_name,
            # HAPT: Labels 1 to 12 reference activities, label 0 references no activity.
            # RealWorld: Labels 0 to 7, label 0 is an activity.
            'number_of_classes': 13 if self.dataset_name == 'hapt' else 8,
            'feature_width': 6, # 2 sensors, each 3 axis.
            'window_size': self.window_size,
            'window_shift': self.window_shift,
            'sequence_to_label': self.sequence_to_label
        }

        if PRINT_DATASET_DISTIBUTION:
            self._get_class_distribution(training_dataset, validation_dataset, test_dataset, dataset_info['number_of_classes'] )
            exit()

        return training_dataset, validation_dataset, test_dataset, dataset_info


    def _get_class_distribution (self, training_dataset, validation_dataset, test_dataset, number_of_classes):
                
        def count_labels (counts, labels, number_of_classes=number_of_classes):
            for label in range(number_of_classes):
                count = tf.cast(labels == label, tf.int32)
                counts[label] += tf.reduce_sum(count)
            return counts
        
        dataset_names = ('training', 'validation', 'test')
        datasets = (training_dataset, validation_dataset, test_dataset)

        dataset_counts = list()
      
        for dataset_name, dataset in zip(dataset_names, datasets):
            
            if self.sequence_to_label:
                labels = dataset.map(lambda feature_window, labels: labels)
            else:
                labels = dataset.flat_map(lambda feature_window, labels: tf.data.Dataset.from_tensor_slices(labels) )            
            
            initial_state = {i: 0 for i in range(number_of_classes) }
            label_counts = labels.reduce(initial_state=initial_state, reduce_func=count_labels)

            types = list(label_counts.keys() )
            counts = [int(v) for v in label_counts.values() ]
            count_sum = sum(counts)            
            percentages = [count / count_sum for count in counts]
            formatted_percentages = [f'{percentage:.2f}' for percentage in percentages]
            formatted_percentages = '[' + ', '.join(formatted_percentages) + ']'
            logging.info('Distributions:\n'
                        f"Dataset: '{dataset_name}\n"
                        f'Labels: {types}\n'
                        f'Counts: {counts}\n'
                        f'Percentages: {formatted_percentages}'
                        f'Count: {count_sum}')
            
            dataset_counts.append(count_sum)

            if self.dataset_name == 'hapt':
                count_sum = sum(counts[1:] )
                percentages = [count / count_sum for count in counts[1:] ]
                formatted_percentages = [f'{percentage:.2f}' for percentage in percentages]
                formatted_percentages = '[' + ', '.join(formatted_percentages) + ']'
                logging.info(f'Distributions without label 0:\n'
                             f'Percentages: {formatted_percentages}')

        combined_count = sum(dataset_counts)
        logging.info(f'Number of samples in dataset: {combined_count}')


    @staticmethod
    def get_dataset_size (dataset) -> int:
        return len(list(dataset.as_numpy_iterator() ) )


    def _create_rw_dataset (self) -> None:

        training_dataset, validation_dataset, test_dataset = list(), list(), list()

        # Which probands make up the datasets:
        training_dataset_probands =   (1, 2, 5, 8, 11, 12, 13, 15)        
        test_dataset_probands =       (9, 10)
        # validation_dataset_proband = 3

        # Label numbers are alphabetically selected.
        activity_names = ('climbingdown', 'climbingup', 'jumping', 'lying', 'running', 'sitting', 'standing', 'walking')
        labels = {activity_name:index for activity_name, index in zip(activity_names, range(len(activity_names) ) ) }
        
        proband_data_directories = [content / 'data' for content in self.dataset_directory.iterdir()
                                        if content.is_dir() and content.name.startswith('proband') ]

        for proband_data_directory in proband_data_directories:
            # Get the proband number from the directory name.
            proband_number = int(re.findall('[0-9]+', proband_data_directory.parent.name)[0] )
            
            if proband_number in (4, 6, 7, 14): # The data from these probands should not be used.
                continue
            
            # Select dataset from proband number.
            dataset = training_dataset if proband_number in training_dataset_probands \
                        else test_dataset if proband_number in test_dataset_probands \
                        else validation_dataset
            
            # Get the paths for directories containing accelerometer data. CSV and SQL files are available, CSV is used.
            zipped_accelerometer_directories = [directory for directory in proband_data_directory.iterdir()
                if directory.name.startswith('acc_') and directory.stem.endswith('_csv') and directory.suffix == '.zip']
                        
            for zipped_accelerometer_directory in zipped_accelerometer_directories:
                # Get label from activity name and activity name from directory name
                label = labels[zipped_accelerometer_directory.stem.split('_')[1] ]

                data = self._load_rw_sensor_data(zipped_accelerometer_directory)

                dataset.append( (
                    data,
                    np.full(shape=(data.shape[0], 1), fill_value=label, dtype=np.uint8) # To keep the different datasets consistend.
                 ) )
        
        logging.info(f'Datasets created; Dataset sizes: Training: {len(training_dataset) }, Validation: {len(validation_dataset) }, Test: {len(test_dataset) }')

        self._write_tfrecords(training_dataset, validation_dataset, test_dataset)
    

    def _load_rw_sensor_data (self, zipped_accelerometer_directory: pathlib.Path) -> pd.DataFrame:
        # Get accelerometer data.
        with zipfile.ZipFile(file=zipped_accelerometer_directory) as zipped_accelerometer_directory_zipfile:
            zipped_accelerometer_file = next( (file_name for file_name in zipped_accelerometer_directory_zipfile.namelist()
                                                        if file_name.endswith(f'_{self.sensor_position}.csv') ) )
            with zipped_accelerometer_directory_zipfile.open(zipped_accelerometer_file) as accelerometer_file:
                accelerometer_data = pd.read_csv(filepath_or_buffer=accelerometer_file)
                accelerometer_data = accelerometer_data.filter(like='attr') # Remove 'id' column.

        # Matching gyroscope file for the same activity, partition returns: '<activity_name>_csv'
        zipped_gyroscope_directory = zipped_accelerometer_directory.parent / ('gyr_' + zipped_accelerometer_directory.name.partition('_')[2] )

        # Get gyroscope data.
        with zipfile.ZipFile(file=zipped_gyroscope_directory) as zipped_gyroscope_directory_zipfile:
            zipped_gyroscope_file = next( (file_name for file_name in zipped_gyroscope_directory_zipfile.namelist() 
                                                    if file_name.endswith(f'_{self.sensor_position}.csv') ) )
            
            with zipped_gyroscope_directory_zipfile.open(zipped_gyroscope_file) as gyroscope_file:
                gyroscope_data = pd.read_csv(filepath_or_buffer=gyroscope_file)
                gyroscope_data = gyroscope_data.filter(like='attr') # Remove 'id' column.
            
        # Join the two tables with the shorter one as the primary one, dropping the unused data of the longer one.
        joined_data = pd.merge_asof(left=gyroscope_data , right=accelerometer_data, on='attr_time', direction='nearest') \
                        if accelerometer_data.shape[0] > gyroscope_data.shape[0] else \
                        pd.merge_asof(left=accelerometer_data , right=gyroscope_data, on='attr_time', direction='nearest')
        
        # Remove the first and last 5 seconds of data, because of noise.
        # Time in Unix-time in milliseconds
        time_column_name = joined_data.columns[0]
        after_first_five_seconds = joined_data[time_column_name] > (joined_data[time_column_name].iat[0]  + 5000)
        before_last_five_seconds = joined_data[time_column_name] < (joined_data[time_column_name].iat[-1] - 5000)
        joined_data = joined_data[after_first_five_seconds & before_last_five_seconds]
        
        return scipy.stats.zscore(joined_data.iloc[:, 1:7] ) # Remove 'time' column, and compute z-score.


    def _create_hapt_dataset (self) -> None:
        # Load label file
        data_directory = self.dataset_directory / 'RawData'
        label_file_path = data_directory / 'labels.txt'
        raw_label_data = np.loadtxt(fname=label_file_path, dtype=int)

        # Label file contains labels for each experiment in one list,
        # the list has to be split into sections, one section per experiment.
        label_sections = list()
        for value, index, count in zip(*np.unique(raw_label_data[:,:2], axis=0, return_counts=True, return_index=True) ):
            label_sections.append( {
                'experiment': value[0],
                'user': value[1],
                'data': raw_label_data[index: index+count, 2:]
            } )

        logging.info('Loaded label file: Number of label sections: ' + str(len(label_sections) ) )


        accelerometer_files, gyroscope_files = list(), list()

        # Gather info. from file names
        # File names have the structure: <sensor-type>_exp<experiment-number>_user<user-number>.txt
        for sensor_data_file_path in [path for path in data_directory.iterdir() if path.name != 'labels.txt']:
            sensor, experiment, user = sensor_data_file_path.stem.split(sep='_')

            if sensor == 'acc':
                accelerometer_files.append( {
                'sensor': sensor,
                'experiment': int(experiment[3:] ),
                'user': int(user[4:] ),
                'file_path': sensor_data_file_path
                } )
            elif sensor == 'gyro':
                gyroscope_files.append( {
                'sensor': sensor,
                'experiment': int(experiment[3:] ),
                'user': int(user[4:] ),
                'file_path': sensor_data_file_path
                } )
            else:
                logging.info("Unknown file in 'rawData' directory:", sensor_data_file_path)
                continue

        logging.info('Number of accelerometer-data files: ' + str(len(accelerometer_files) ) )
        logging.info('Number of gyroscope-data files: ' + str(len(gyroscope_files) ) )


        # Create datasets
        train_dataset, validation_dataset, test_dataset = list(), list(), list()

        # User-numbers are used to split the dataset approx.: (70/10/20)%
        training_user_numbers =   range( 1, 22)
        validation_user_numbers = range(28, 31)
        test_user_numbers =       range(22, 28)

        for accelerometer_file in accelerometer_files:
            # Find file containing the data of the gyroscope sensor of the same experiment
            gyroscope_file = next(gyroscope_file for gyroscope_file in gyroscope_files 
                                    if gyroscope_file['experiment'] == accelerometer_file['experiment'] )
                
            label_data = next(label_section['data'] for label_section in label_sections
                                if label_section['experiment'] == accelerometer_file['experiment'] )

            # Load sensor data file and normalise the data per channel then combine the data into one array.
            sensor_data = np.hstack( (
                scipy.stats.zscore(np.loadtxt(fname=accelerometer_file['file_path'], dtype=np.float64), axis=0),
                scipy.stats.zscore(np.loadtxt(fname=gyroscope_file['file_path'], dtype=np.float64), axis=0)                
            ) )

            # Create label sequence; some sections are not labeled, these stay labeled zero.
            labels = np.zeros(shape=(sensor_data.shape[0], 1), dtype=np.uint8)
            for label, start, end in label_data:
                labels[start-1: end] = label
            
            # Split the data into the datasets
            if accelerometer_file['user'] in training_user_numbers:
                train_dataset.append( (sensor_data, labels) )
            elif accelerometer_file['user'] in validation_user_numbers:
                validation_dataset.append( (sensor_data, labels) )
            elif accelerometer_file['user'] in test_user_numbers:
                test_dataset.append( (sensor_data, labels) )

        logging.info(f'Datasets created; Dataset sizes: Training: {len(train_dataset) }, Validation: {len(validation_dataset) }, Test: {len(test_dataset) }')

        self._write_tfrecords(train_dataset, validation_dataset, test_dataset)


    def _write_tfrecords (self, train_dataset, validation_dataset, test_dataset) -> None:
        # The numpy arrays are serialised to preserve the shape of the data.
        # Tensorflow only accepts row arrays as features, an alternative would be to flatten the arrays and add
        # the original shape as a feature, so that the shape can be recreated after parsing the data.
        
        def _create_feature (data) -> tf.train.Feature:
            return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(data).numpy() ] ) )

        def _serialise_dataset (dataset) -> typing.List[bytes]:
            serialised_dataset = list()
            for element in dataset:
                feature = {
                    'sensor_data': _create_feature(element[0] ),
                    'label':       _create_feature(element[1] )
                }

                example = tf.train.Example(features=tf.train.Features(feature=feature) )
                serialised_dataset.append(example.SerializeToString() )
            
            return serialised_dataset

        train_dataset_serialised =      _serialise_dataset(train_dataset)
        validation_dataset_serialised = _serialise_dataset(validation_dataset)
        test_dataset_serialised =       _serialise_dataset(test_dataset)

        def _write_tfrecords_files (file_name, data) -> None:
            with tf.io.TFRecordWriter(path=str(self.tfrecords_directory / file_name) ) as tfrecord_writer:
                for element in data:
                    tfrecord_writer.write(element)

        _write_tfrecords_files('train.tfrecords', train_dataset_serialised)
        _write_tfrecords_files('validation.tfrecords', validation_dataset_serialised)
        _write_tfrecords_files('test.tfrecords', test_dataset_serialised)

        logging.info(f"TFRecords created in: '{self.tfrecords_directory}'.")


    def _load_tfrecords (self) -> typing.Tuple[tf.data.TFRecordDataset, tf.data.TFRecordDataset, tf.data.TFRecordDataset]:
        global PRINT_DATASET_SIZE
        
        logging.info(f"Reading TFRecords from: '{self.tfrecords_directory}'")
        
        training_dataset =   tf.data.TFRecordDataset(filenames=str(self.tfrecords_directory / 'train.tfrecords') )
        validation_dataset = tf.data.TFRecordDataset(filenames=str(self.tfrecords_directory / 'validation.tfrecords') )
        test_dataset =       tf.data.TFRecordDataset(filenames=str(self.tfrecords_directory / 'test.tfrecords') )


        def _parse_serialised_dataset (serialised_data) -> tf.Tensor:
            # The features are strings because the arrays are serialised.
            feature_description = {
                'sensor_data': tf.io.FixedLenFeature(shape=[], dtype=tf.string),
                'label':       tf.io.FixedLenFeature(shape=[], dtype=tf.string)
            }
            parsed_data = tf.io.parse_single_example(serialised_data, feature_description)
            sensor_data = tf.io.parse_tensor(serialized=parsed_data['sensor_data'], out_type=tf.float64)
            label =       tf.io.parse_tensor(serialized=parsed_data['label'], out_type=tf.uint8)
            label = tf.cast(label, dtype=np.float64) # Cast label to the same type as sensor_data to make joining of arrays possible.
            return tf.concat( (sensor_data, label), axis=1) # Join sensor_data and labels to make windowing easier.

        training_dataset =   training_dataset.map(map_func=_parse_serialised_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        validation_dataset = validation_dataset.map(map_func=_parse_serialised_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        test_dataset =       test_dataset.map(map_func=_parse_serialised_dataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)

        if PRINT_DATASET_SIZE:
            logging.info('Counting dataset...')
            size_of_training_dataset =   self.get_dataset_size(training_dataset.flat_map(tf.data.Dataset.from_tensor_slices) )
            size_of_validation_dataset = self.get_dataset_size(validation_dataset.flat_map(tf.data.Dataset.from_tensor_slices) )
            size_of_test_dataset =       self.get_dataset_size(test_dataset.flat_map(tf.data.Dataset.from_tensor_slices) )

            size_of_dataset = size_of_training_dataset + size_of_validation_dataset + size_of_test_dataset

            logging.info(f"Size of the dataset '{self.dataset_name}'\n"
                         f'Training dataset: {size_of_training_dataset:,}\n'
                         f'Validation dataset: {size_of_validation_dataset:,}\n'
                         f'Test dataset: {size_of_test_dataset:,}\n'
                         f'Combinded size: {size_of_dataset:,}')
            
            exit()


        def _create_windows (data) -> tf.data.Dataset:
            data = tf.data.Dataset.from_tensor_slices(data)
            data = data.window(self.window_size, shift=self.window_shift, stride=1, drop_remainder=True)

            # window() creates dataset of sub-datasets, batching each sub-dataset/window by the size of a window
            # and flattening the result creates a dataset of tensors.
            data = data.flat_map(lambda window: window.batch(self.window_size) )

            # Seperate feature and label tensors
            return data.map(lambda element: (element[:,:-1], tf.cast(element[:,-1:], dtype=np.uint8) ) )

        training_dataset =   training_dataset.flat_map(lambda sequence: _create_windows(sequence) )
        validation_dataset = validation_dataset.flat_map(lambda sequence: _create_windows(sequence) )
        test_dataset =       test_dataset.flat_map(lambda sequence: _create_windows(sequence) )
        logging.info('Datasets windowed.')
        
        if self.sequence_to_label:
            # Only keep windows where all labels are of the same class.
            # Reduce label vectors, containing the same value in all elements, to a skalar.
            # Label is a tensor of vector tensors of size one.

            if self.dataset_name == 'rw': # In the RealWorld dataset all elements only contain one label.
                training_dataset =   training_dataset.map(lambda feature, label: (feature, label[0][0] ) )
                validation_dataset = validation_dataset.map(lambda feature, label: (feature, label[0][0] ) )                
                test_dataset =       test_dataset.map(lambda feature, label: (feature, label[0][0] ) )
            
            else:
                def _single_unique_label (features, labels) -> tf.Tensor:
                    flattened_labels = tf.reshape(labels, shape=(1, self.window_size) ) # Labels are column vectors
                    flattened_labels = tf.squeeze(flattened_labels)
                    unique_labels, indices = tf.unique(flattened_labels)
                    number_of_unique_labels = tf.shape(unique_labels)[0] # Test if the number of unique labels is one
                    return number_of_unique_labels == 1

                training_dataset = training_dataset.filter(_single_unique_label)
                training_dataset = training_dataset.map(lambda feature, label: (feature, label[0][0] ) )

                validation_dataset = validation_dataset.filter(_single_unique_label)
                validation_dataset = validation_dataset.map(lambda feature, label: (feature, label[0][0] ) )
                
                test_dataset = test_dataset.filter(_single_unique_label)
                test_dataset = test_dataset.map(lambda feature, label: (feature, label[0][0] ) )
        
        return training_dataset, validation_dataset, test_dataset
