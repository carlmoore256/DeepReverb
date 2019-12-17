import keras
import numpy as np
from model_file import ModelFile
from keras.preprocessing.image import ImageDataGenerator

class Predict:

    def PredictResults(model, dataloader):
        # create the flow from directory objects
        test_generator = dataloader.TestGen()
        train_generator = dataloader.TrainGen()
        test_steps = len(test_generator.filenames)

        # run the predict generator
        predictions = model.predict_generator(test_generator, steps=test_steps, verbose=1)

        # write the results to a file
        predicted_class_indices=np.argmax(predictions,axis=1)
        labels = (train_generator.class_indices)
        labels = dict((v,k) for k,v in labels.items())
        predictions = [labels[k] for k in predicted_class_indices]
        filenames=test_generator.filenames
        ModelFile.SavePredictions(filenames,predictions)
