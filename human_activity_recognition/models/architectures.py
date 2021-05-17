import gin
import tensorflow as tf


def load_models(model_config, dataset_info):
    """
    Instantiate an model from the model_config
    Parameters
    ----------
        model_config (dict): input shape of the neural network
        dataset_info (dict): number of classes, corresponding to the number of output neurons
    Returns
    ----------
        (list): list of keras.Model

    """
    model_types = ['simple_rnn', 'simple_lstm', 'simple_gru',
                   'bidirectional_rnn', 'bidirectional_lstm', 'bidirectional_gru',
                   'stacked_rnn', 'stacked_lstm', 'stacked_gru']

    # Checking for valid models
    if isinstance(model_config, dict):
        model = model_config['model']
        if model not in model_types:
            raise ValueError("Invalid model name.")
        model_config = [model_config]  # Convert dict to list for easier processing.
    elif isinstance(model_config, list):
        if any([model['model'] not in model_types for model in model_config]):
            raise ValueError("Invalid model name.")
    else:
        raise ValueError("Invalid model name.")

    models = list()
    input_shape = (dataset_info['window_size'], dataset_info['feature_width'])
    number_of_classes = dataset_info['number_of_classes']
    for model_dict in model_config:
        model_name = model_dict['model']
        if model_name == 'simple_rnn':
            units = None
            activation = None
            if 'units' in model_dict:
                units = model_dict['units']
            if 'activation' in model_dict:
                activation = model_dict['activation']
            models.append((model_name,
                           simple_rnn(input_shape=input_shape,
                                      number_of_classes=number_of_classes,
                                      sequence_to_label=dataset_info["sequence_to_label"],
                                      units=units,
                                      activation=activation)))

        elif model_name == 'simple_lstm':
            units = None
            activation = None
            if 'units' in model_dict:
                units = model_dict['units']
            if 'activation' in model_dict:
                activation = model_dict['activation']
            models.append((model_name,
                           simple_lstm(input_shape=input_shape,
                                       number_of_classes=number_of_classes,
                                       sequence_to_label=dataset_info["sequence_to_label"],
                                       units=units,
                                       activation=activation)))

        elif model_name == 'simple_gru':
            units = None
            activation = None
            if 'units' in model_dict:
                units = model_dict['units']
            if 'activation' in model_dict:
                activation = model_dict['activation']
            models.append((model_name,
                           simple_gru(input_shape=input_shape,
                                      number_of_classes=number_of_classes,
                                      sequence_to_label=dataset_info["sequence_to_label"],
                                      units=units,
                                      activation=activation)))

        elif model_name == 'bidirectional_rnn':
            if not dataset_info["sequence_to_label"]:
                units = None
                activation = None
                if 'units' in model_dict:
                    units = model_dict['units']
                if 'activation' in model_dict:
                    activation = model_dict['activation']
                models.append((model_name,
                               bidirectional_rnn(input_shape=input_shape,
                                                 number_of_classes=number_of_classes,
                                                 sequence_to_label=dataset_info["sequence_to_label"],
                                                 units=units,
                                                 activation=activation)))
            else:
                raise ValueError("No Sequence to Label classification for Bidirectional RNN")

        elif model_name == 'bidirectional_lstm':
            if not dataset_info["sequence_to_label"]:
                units = None
                activation = None
                if 'units' in model_dict:
                    units = model_dict['units']
                if 'activation' in model_dict:
                    activation = model_dict['activation']
                models.append((model_name,
                               bidirectional_lstm(input_shape=input_shape,
                                                  number_of_classes=number_of_classes,
                                                  sequence_to_label=dataset_info["sequence_to_label"],
                                                  units=units,
                                                  activation=activation)))
            else:
                raise ValueError("No Sequence to Label classification for Bidirectional LSTM")

        elif model_name == 'bidirectional_gru':
            if not dataset_info["sequence_to_label"]:
                units = None
                activation = None
                if 'units' in model_dict:
                    units = model_dict['units']
                if 'activation' in model_dict:
                    activation = model_dict['activation']
                models.append((model_name,
                               bidirectional_gru(input_shape=input_shape,
                                                 number_of_classes=number_of_classes,
                                                 sequence_to_label=dataset_info["sequence_to_label"],
                                                 units=units,
                                                 activation=activation)))
            else:
                raise ValueError("No Sequence to Label classification for Bidirectional GRU")

        elif model_name == 'stacked_rnn':
            units = None
            activation = None
            if 'units' in model_dict:
                units = model_dict['units']
            if 'activation' in model_dict:
                activation = model_dict['activation']
            models.append((model_name,
                           stacked_rnn(input_shape=input_shape,
                                       number_of_classes=number_of_classes,
                                       sequence_to_label=dataset_info["sequence_to_label"],
                                       units=units,
                                       activation=activation)))

        elif model_name == 'stacked_lstm':
            units = None
            activation = None
            if 'units' in model_dict:
                units = model_dict['units']
            if 'activation' in model_dict:
                activation = model_dict['activation']
            models.append((model_name,
                           stacked_lstm(input_shape=input_shape,
                                        number_of_classes=number_of_classes,
                                        sequence_to_label=dataset_info["sequence_to_label"],
                                        units=units,
                                        activation=activation)))

        elif model_name == 'stacked_gru':
            units = None
            activation = None
            if 'units' in model_dict:
                units = model_dict['units']
            if 'activation' in model_dict:
                activation = model_dict['activation']
            models.append((model_name,
                           stacked_gru(input_shape=input_shape,
                                       number_of_classes=number_of_classes,
                                       sequence_to_label=dataset_info["sequence_to_label"],
                                       units=units,
                                       activation=activation)))

    return models


def simple_rnn(input_shape, number_of_classes, sequence_to_label=True, **kwargs):
    """Defines a SimpleRNN architecture.

    Parameters
    ----------
        `input_shape` (tuple: 3): input shape of the neural network
        `number_of_classes` (int): number of classes, corresponding to the number of output neurons
        `sequence_to_label` (boolean): determines the type of output
            output -> sequence to label if True
            output -> sequence to sequence if False
    Returns
    ----------
        (keras.Model): keras model object
    """

    # units: Positive integer, dimensionality of the output space of the RN
    units = 20  #

    # Activation function to use. Default: hyperbolic tangent (tanh).
    # If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).
    activation = 'tanh'
    for key, value in kwargs.items():
        if key == 'units' and value is not None and isinstance(value, int):
            units = value
            assert units > 0, 'Number of units has to be at least 1.'
        if key == 'activation' and value is not None and value != "tanh":
            activation = value

    inputs = tf.keras.Input(input_shape)
    if not sequence_to_label:
        simple_rnn = tf.keras.layers.SimpleRNN(units=units, return_sequences=True, activation=activation)(inputs)
        output_logits = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(number_of_classes), name='output_logits')(simple_rnn)
        output_prob = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Softmax())(output_logits)
    else:
        simple_rnn = tf.keras.layers.SimpleRNN(units=units, activation=activation)(inputs)
        output_logits = tf.keras.layers.Dense(number_of_classes, name='output_logits')(simple_rnn)
        output_prob = tf.keras.layers.Softmax()(output_logits)

    name = '-'.join(['simple_rnn', str(units), activation])

    return tf.keras.Model(inputs=inputs, outputs=output_prob, name=name)


def simple_lstm(input_shape, number_of_classes, sequence_to_label=True, **kwargs):
    """Defines a block of LSTM architecture.

        Parameters
        ----------
            `input_shape` (tuple: 3): input shape of the neural network
            `number_of_classes` (int): number of classes, corresponding to the number of output neurons
            `sequence_to_label` (boolean): determines the type of output
                output -> sequence to label if True
                output -> sequence to sequence if False
        Returns
        ----------
            (keras.Model): keras model object
        """
    # units: Positive integer, dimensionality of the output space of the RN
    units = 20  #

    # Activation function to use. Default: hyperbolic tangent (tanh).
    # If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).
    activation = 'tanh'
    for key, value in kwargs.items():
        if key == 'units' and value is not None and isinstance(value, int):
            units = value
            assert units > 0, 'Number of units has to be at least 1.'
        if key == 'activation' and value is not None and value != "tanh":
            activation = value

    inputs = tf.keras.Input(input_shape)
    if not sequence_to_label:
        simple_lstm = tf.keras.layers.LSTM(units=units, return_sequences=True, activation=activation)(inputs)
        output_logits = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(number_of_classes), name='output_logits')(simple_lstm)
        output_prob = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Softmax())(output_logits)
    else:
        simple_lstm = tf.keras.layers.LSTM(units=units, activation=activation)(inputs)
        output_logits = tf.keras.layers.Dense(number_of_classes, name='output_logits')(simple_lstm)
        output_prob = tf.keras.layers.Softmax()(output_logits)

    name = '-'.join(['simple_lstm', str(units), activation])

    return tf.keras.Model(inputs=inputs, outputs=output_prob, name=name)


def simple_gru(input_shape, number_of_classes, sequence_to_label=True, **kwargs):
    """Defines a block of GRU architecture.

        Parameters
        ----------
            `input_shape` (tuple: 3): input shape of the neural network
            `number_of_classes` (int): number of classes, corresponding to the number of output neurons
            `sequence_to_label` (boolean): determines the type of output
                output -> sequence to label if True
                output -> sequence to sequence if False
        Returns
        ----------
            (keras.Model): keras model object
    """
    # units: Positive integer, dimensionality of the output space of the RN
    units = 20  #

    # Activation function to use. Default: hyperbolic tangent (tanh).
    # If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).
    activation = 'tanh'
    for key, value in kwargs.items():
        if key == 'units' and value is not None and isinstance(value, int):
            units = value
            assert units > 0, 'Number of units has to be at least 1.'
        if key == 'activation' and value is not None and value != "tanh":
            activation = value

    inputs = tf.keras.Input(input_shape)

    if not sequence_to_label:
        simple_gru = tf.keras.layers.GRU(units=units, return_sequences=True, activation=activation)(inputs)
        output_logits = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(number_of_classes), name='output_logits')(simple_gru)
        output_prob = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Softmax())(output_logits)
    else:
        simple_gru = tf.keras.layers.GRU(units=units, activation=activation)(inputs)
        output_logits = tf.keras.layers.Dense(number_of_classes, name='output_logits')(simple_gru)
        output_prob = tf.keras.layers.Softmax()(output_logits)

    name = '-'.join(['simple_gru', str(units), activation])

    return tf.keras.Model(inputs=inputs, outputs=output_prob, name=name)


def bidirectional_rnn(input_shape, number_of_classes, sequence_to_label=True, **kwargs):
    """Defines a Bidirection SimpleRNN architecture.

        Parameters
        ----------
            `input_shape` (tuple: 3): input shape of the neural network
            `number_of_classes` (int): number of classes, corresponding to the number of output neurons
            `sequence_to_label` (boolean): determines the type of output
                output -> sequence to label if True
                output -> sequence to sequence if False
        Returns
        ----------
            (keras.Model): keras model object
    """

    # units: Positive integer, dimensionality of the output space of the RN
    units = 20  #

    # Activation function to use. Default: hyperbolic tangent (tanh).
    # If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).
    activation = 'tanh'
    for key, value in kwargs.items():
        if key == 'units' and value is not None and isinstance(value, int):
            units = value
            assert units > 0, 'Number of units has to be at least 1.'
        if key == 'activation' and value is not None and value != "tanh":
            activation = value

    inputs = tf.keras.Input(input_shape)
    simple_rnn = tf.keras.layers.SimpleRNN(units=units, return_sequences=True, activation=activation)
    bidirectional_simple_rnn = tf.keras.layers.Bidirectional(simple_rnn)(inputs)
    if not sequence_to_label:
        output_logits = tf.keras.layers.Dense(number_of_classes, name='output_logits')(bidirectional_simple_rnn)
        output_prob = tf.keras.layers.Softmax()(output_logits)
    else:
        raise ValueError("No Sequence to Label classification for Bidirectional RNN")

    name = '-'.join(['bidirectional_rnn', str(units), activation])

    return tf.keras.Model(inputs=inputs, outputs=output_prob, name=name)


def bidirectional_lstm(input_shape, number_of_classes, sequence_to_label=True, **kwargs):
    """Defines a Bidirection LSTM architecture.

        Parameters
        ----------
            `input_shape` (tuple: 3): input shape of the neural network
            `number_of_classes` (int): number of classes, corresponding to the number of output neurons
            `sequence_to_label` (boolean): determines the type of output
                output -> sequence to label if True
                output -> sequence to sequence if False
        Returns
        ----------
            (keras.Model): keras model object
    """

    # units: Positive integer, dimensionality of the output space of the RN
    units = 20  #

    # Activation function to use. Default: hyperbolic tangent (tanh).
    # If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).
    activation = 'tanh'
    for key, value in kwargs.items():
        if key == 'units' and value is not None and isinstance(value, int):
            units = value
            assert units > 0, 'Number of units has to be at least 1.'
        if key == 'activation' and value is not None and value != "tanh":
            activation = value

    inputs = tf.keras.Input(input_shape)
    simple_lstm = tf.keras.layers.LSTM(units=units, return_sequences=True, activation=activation)
    bidirectional_simple_lstm = tf.keras.layers.Bidirectional(simple_lstm)(inputs)
    if not sequence_to_label:
        output_logits = tf.keras.layers.Dense(number_of_classes, name='output_logits')(bidirectional_simple_lstm)
        output_prob = tf.keras.layers.Softmax()(output_logits)
    else:
        return ValueError("No Sequence to Label classification for Bidirectional LSTM")

    name = '-'.join(['bidirectional_lstm', str(units), activation])

    return tf.keras.Model(inputs=inputs, outputs=output_prob, name=name)


def bidirectional_gru(input_shape, number_of_classes, sequence_to_label=True, **kwargs):
    """Defines a Bidirection GRU architecture.

        Parameters
        ----------
            `input_shape` (tuple: 3): input shape of the neural network
            `number_of_classes` (int): number of classes, corresponding to the number of output neurons
            `sequence_to_label` (boolean): determines the type of output
                output -> sequence to label if True
                output -> sequence to sequence if False
        Returns
        ----------
            (keras.Model): keras model object
    """

    # units: Positive integer, dimensionality of the output space of the RN
    units = 20  #

    # Activation function to use. Default: hyperbolic tangent (tanh).
    # If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).
    activation = 'tanh'
    for key, value in kwargs.items():
        if key == 'units' and value is not None and isinstance(value, int):
            units = value
            assert units > 0, 'Number of units has to be at least 1.'
        if key == 'activation' and value is not None and value != "tanh":
            activation = value

    inputs = tf.keras.Input(input_shape)
    simple_gru = tf.keras.layers.GRU(units=units, return_sequences=True, activation=activation)
    bidirectional_simple_gru = tf.keras.layers.Bidirectional(simple_gru)(inputs)
    if not sequence_to_label:
        output_logits = tf.keras.layers.Dense(number_of_classes, name='output_logits')(bidirectional_simple_gru)
        output_prob = tf.keras.layers.Softmax()(output_logits)
    else:
        return ValueError("No Sequence to Label classification for Bidirectional GRU")

    name = '-'.join(['bidirectional_gru', str(units), activation])

    return tf.keras.Model(inputs=inputs, outputs=output_prob, name=name)


def stacked_rnn(input_shape, number_of_classes, sequence_to_label=True, **kwargs):
    """Defines a Stack of RNNs architecture.

       Parameters
       ----------
           `input_shape` (tuple: 3): input shape of the neural network
           `number_of_classes` (int): number of classes, corresponding to the number of output neurons
           `sequence_to_label` (boolean): determines the type of output
               output -> sequence to label if True
               output -> sequence to sequence if False
       Returns
       ----------
           (keras.Model): keras model object
       """

    # units: list of Positive integer, dimensionality of the output space of the RNN in each stack
    units = [20, 10]

    # Activation function to use. Default: hyperbolic tangent (tanh).
    # If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).
    activation = 'tanh'
    for key, value in kwargs.items():
        if key == 'units' and value is not None and isinstance(value, list):
            if len(units) >= 2 and all(isinstance(u, int) for u in units):
                units = value
                assert all(u > 0 for u in units), 'Number of units has to be at least 1.'
        if key == 'activation' and value is not None and value != "tanh":
            activation = value

    inputs = tf.keras.Input(input_shape)
    stack_n = tf.keras.layers.SimpleRNN(units=units[0], return_sequences=True, activation=activation)(inputs)

    for u in units[1:-1]:
        stack_n = tf.keras.layers.SimpleRNN(units=u, return_sequences=True, activation=activation)(stack_n)

    if not sequence_to_label:
        stack_last = tf.keras.layers.SimpleRNN(units=units[-1], return_sequences=True, activation=activation)(stack_n)
        outputs = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(number_of_classes, activation="softmax"))(stack_last)
    else:
        stack_last = tf.keras.layers.SimpleRNN(units=units[-1], activation=activation)(stack_n)
        outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(stack_last)

    units_str = [str(u) for u in units]
    units_str = "_".join(units_str)
    name = '-'.join(['stack_rnn', units_str, activation])

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def stacked_lstm(input_shape, number_of_classes, sequence_to_label=True, **kwargs):
    """Defines a Stack of RNNs architecture.

	   Parameters
	   ----------
		   `input_shape` (tuple: 3): input shape of the neural network
		   `number_of_classes` (int): number of classes, corresponding to the number of output neurons
		   `sequence_to_label` (boolean): determines the type of output
			   output -> sequence to label if True
			   output -> sequence to sequence if False
	   Returns
	   ----------
		   (keras.Model): keras model object
	   """

    # units: list of Positive integer, dimensionality of the output space of the RNN in each stack

    units = [20, 10]

    # Activation function to use. Default: hyperbolic tangent (tanh).
    # If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).
    activation = 'tanh'
    for key, value in kwargs.items():
        if key == 'units' and value is not None and isinstance(value, list):
            if len(units) >= 2 and all(isinstance(u, int) for u in units):
                units = value
                assert all(u > 0 for u in units), 'Number of units has to be at least 1.'
        if key == 'activation' and value is not None and value != "tanh":
            activation = value

    inputs = tf.keras.Input(input_shape)
    stack_n = tf.keras.layers.LSTM(units=units[0], return_sequences=True, activation=activation)(inputs)

    for u in units[1:-1]:
        stack_n = tf.keras.layers.LSTM(units=u, return_sequences=True, activation=activation)(stack_n)

    if not sequence_to_label:
        stack_last = tf.keras.layers.LSTM(units=units[-1], return_sequences=True, activation=activation)(stack_n)
        outputs = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(number_of_classes, activation="softmax"))(stack_last)
    else:
        stack_last = tf.keras.layers.LSTM(units=units[-1], activation=activation)(stack_n)
        outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(stack_last)

    units_str = [str(u) for u in units]
    units_str = "_".join(units_str)
    name = '-'.join(['stack_lstm', units_str, activation])

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)


def stacked_gru(input_shape, number_of_classes, sequence_to_label=True, **kwargs):
    """Defines a Stack of RNNs architecture.

	   Parameters
	   ----------
		   `input_shape` (tuple: 3): input shape of the neural network
		   `number_of_classes` (int): number of classes, corresponding to the number of output neurons
		   `sequence_to_label` (boolean): determines the type of output
			   output -> sequence to label if True
			   output -> sequence to sequence if False
	   Returns
	   ----------
		   (keras.Model): keras model object
	   """

    # units: list of Positive integer, dimensionality of the output space of the RNN in each stack
    units = [20, 10]

    # Activation function to use. Default: hyperbolic tangent (tanh).
    # If you pass None, no activation is applied (ie. "linear" activation: a(x) = x).
    activation = 'tanh'
    for key, value in kwargs.items():
        if key == 'units' and value is not None and isinstance(value, list):
            if len(units) >= 2 and all(isinstance(u, int) for u in units):
                units = value
                assert all(u > 0 for u in units), 'Number of units has to be at least 1.'
        if key == 'activation' and value is not None and value != "tanh":
            activation = value

    inputs = tf.keras.Input(input_shape)
    stack_n = tf.keras.layers.GRU(units=units[0], return_sequences=True, activation=activation)(inputs)

    for u in units[1:-1]:
        print(f"here = {u}")
        stack_n = tf.keras.layers.GRU(units=u, return_sequences=True, activation=activation)(stack_n)

    if not sequence_to_label:
        stack_last = tf.keras.layers.GRU(units=units[-1], return_sequences=True, activation=activation)(stack_n)
        outputs = tf.keras.layers.TimeDistributed(
            tf.keras.layers.Dense(number_of_classes, activation="softmax"))(stack_last)
    else:
        stack_last = tf.keras.layers.GRU(units=units[-1], activation=activation)(stack_n)
        outputs = tf.keras.layers.Dense(number_of_classes, activation="softmax")(stack_last)

    units_str = [str(u) for u in units]
    units_str = "_".join(units_str)
    name = '-'.join(['stack_gru', units_str, activation])

    return tf.keras.Model(inputs=inputs, outputs=outputs, name=name)
