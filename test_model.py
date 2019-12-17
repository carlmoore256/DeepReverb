from data_loader import LoadData
from evaluate import EvaluateModel
from keras import models
from predict import Predict
import model_file

model_file = 'hpc-result.h5'

img_width, img_height = 256, 256
batch_size = 1

dataloader = LoadData(batch_size, img_width, img_height)
model = models.load_model('./models/' + model_file)
print('evaluating')
eval = EvaluateModel.Evaluate(model,dataloader)
print('making predictions')
predict = Predict.PredictResults(model, dataloader)
