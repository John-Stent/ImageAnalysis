import tensorflow
from keras_preprocessing.image import ImageDataGenerator
import keras
import numpy


def getImageFeaturesAndTrainSet():
    batch_size = 100
    imageDataGenerator = ImageDataGenerator(
        preprocessing_function=tensorflow.keras.applications.resnet50.preprocess_input)
    X_train = imageDataGenerator.flow_from_directory("CorelDB", batch_size=batch_size)

    model = tensorflow.keras.applications.resnet50.ResNet50(include_top=False, weights='imagenet', input_tensor=None,
                                                            input_shape=None, pooling="avg")
    model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    prediction_model = keras.Model(inputs=model.inputs, outputs=model.get_layer(name="avg_pool").output)
    features = numpy.array(prediction_model.predict(X_train))
    return features, X_train
