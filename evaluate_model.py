import keras
from keras.preprocessing.image import ImageDataGenerator

class EvaluateModel:

    def Evaluate(self, model, dataloader):

        datagen = dataloader.(rescale=1./255)

        test_generator = datagen.flow_from_directory(
            directory= test_data,
            target_size=(img_width, img_height),
            color_mode="rgb",
            batch_size=1,
            class_mode='categorical',
            shuffle=False,
            seed=42)
