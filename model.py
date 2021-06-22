import tensorflow as tf
from tensorflow.keras.layers import Conv2D, AveragePooling2D, Flatten, Dense, Input
from tensorflow.keras import Model


def feature_extractor(inputs):
    x = Conv2D(16, activation='relu', kernel_size=3, input_shape=(75, 75, 1))(inputs)
    x = AveragePooling2D((2, 2))(x)

    x = Conv2D(32, activation='relu', kernel_size=3)(x)
    x = AveragePooling2D((2, 2))(x)

    x = Conv2D(64, activation='relu', kernel_size=3)(x)
    x = AveragePooling2D((2, 2))(x)

    return x


def dense_layers(inputs):
    x = Flatten()(inputs)
    x = Dense(128, activation='relu')(x)

    return x


def classifier(inputs):
    classification_output = Dense(10, activation='softmax', name='classification_output')(inputs)
    return classification_output


def bounding_box_regressor(inputs):
    bounding_box_output = Dense(units='4', name="bounding_box_output")(inputs)
    return bounding_box_output


def final_model(inputs):
    feature_cnn = feature_extractor(inputs)
    dense_output = dense_layers(feature_cnn)

    classification_output = classifier(dense_output)
    bounding_box_output = bounding_box_regressor(dense_output)

    model = Model(inputs=inputs, outputs=[classification_output, bounding_box_output])

    return model


def define_and_compile_model(inputs):
    model = final_model(inputs)

    model.compile(optimizer='adam',
                  loss={'classification_output': 'categorical_crossentropy',
                        'bounding_box_output': 'mse'
                        },
                  metrics={'classification_output': 'accuracy',
                           'bounding_box_output': 'mse'
                           })
    return model


def build_model(strategy):
    with strategy.scope():
        inputs = Input(shape=(75, 75, 1))
        model = define_and_compile_model(inputs)

    return model
