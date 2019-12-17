import keras
from base_model import BaseModel
from keras.preprocessing.image import ImageDataGenerator
import model_file

class TrainModel:

    def BeginTraining(epochs, batch_size, num_classes, dataloader, model_file, weight_file, load_weights=False, load_model=False):

        train_gen = dataloader.TrainGen()
        val_gen = dataloader.ValGen()
        x_train, y_train = train_gen.next()

        # either load or create the model
        if load_model:
            model = model_file.LoadModel('/models/' + model_file)
        else:
            model = BaseModel().CreateModel(x_train.shape[1:], num_classes)

        # summarize model.
        model.summary()

        if load_weights:
            model.load_weights('./weights/' + weight_file)

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.SGD(lr=0.02),      #optimizer=keras.optimizers.SGD(lr=0.02)
                      metrics=['accuracy'])

        step_size_train = train_gen.n//train_gen.batch_size
        step_size_valid = val_gen.n//val_gen.batch_size

        model.fit_generator(train_gen, epochs=epochs,
                    steps_per_epoch=step_size_train, validation_data=val_gen,
                     validation_steps=step_size_valid, shuffle=True, verbose=1)

        return model
