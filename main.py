from comet_ml import Experiment
from base_model import BaseModel
from data_loader import LoadData
from train_model import TrainModel
from evaluate import EvaluateModel
from model_file import ModelFile
from predict import Predict
experiment = Experiment('OKhPlin1BVQJFzniHu1f3K1t3')

########### HYPERPARAMETERS ##############
img_width, img_height = 256, 256
num_classes = 7
epochs = 5
batch_size = 128

########### SAVING PARAMETERS ###########
save_model = True
save_weights = True
load_weights = False
weight_file = 'saved_weight-1576127088.591156.h5'
load_model = False
model_file = 'hpc-result.h5'

########### RUN MODEL ###################
# returns a dataloader class containing the data streams
dataloader = LoadData(batch_size, img_width, img_height)
# returns the trained model
model =  TrainModel.BeginTraining(epochs, batch_size, num_classes,
      dataloader, model_file, weight_file, load_weights, load_model)
# evaluates the model
eval = EvaluateModel.Evaluate(model, dataloader)
# make predictions
predict = Predict.PredictResults(model, dataloader)

if save_model:
    ModelFile.SaveModel(model)
if save_weights:
    ModelFile.SaveWeights(model)

print('process complete')
