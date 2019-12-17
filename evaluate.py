import keras
from keras.preprocessing.image import ImageDataGenerator

class EvaluateModel:

    def Evaluate(model, dataloader):

        test_generator = dataloader.TestGen()
        test_steps = len(test_generator.filenames)

        scores = model.evaluate_generator(generator=test_generator, steps=test_steps, verbose=1)
        print('Test loss:', scores[0])
        print('Test accuracy:', scores[1])

        return scores
