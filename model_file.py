import datetime
import numpy as np
import pandas as pd
from keras import models

class ModelFile:

    def SaveModel(model):
        ts = str(datetime.datetime.now().timestamp())
        saved_model = './models/saved_model-' + ts + '.h5'
        model.save(saved_model)
        print('model saved to' + saved_model)

    def SaveWeights(model):
        ts = str(datetime.datetime.now().timestamp())
        weight_file = './weights/saved_weight-' + ts + '.h5'
        model.save_weights(weight_file)
        print('weights saved to' + saved_model)

    def LoadModel(file):
        model = models.load_model('./models/' + file)
        return model

    def SavePredictions(filenames, predictions):
        results=pd.DataFrame({"Filename":filenames, "Predictions":predictions})
        ts = str(datetime.datetime.now().timestamp())
        results.to_csv('./predictions/predictions-'+ ts + '.csv')
