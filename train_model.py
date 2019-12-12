import keras
from base_model import BaseModel
from keras.preprocessing.image import ImageDataGenerator
import data_loader

class TrainModel:

    def BeginTraining(self, epochs, batch_size, img_width, img_height, num_classes, weight_file=None):

        dataloader = data_loader.DataLoader(batch_size, img_width, img_height)
        train_gen = dataloader.TrainGen()
        val_gen = dataloader.ValGen()

        x_train, y_train = train_gen.next()

        model = BaseModel().CreateModel(x_train.shape[1:], num_classes)

        if weight_file != None:
            model.load_weights('./weights/' + weight_file)

        model.compile(loss='categorical_crossentropy',
                      optimizer=keras.optimizers.SGD(lr=0.01),
                      metrics=['accuracy'])

        step_size_train = train_gen.n//train_gen.batch_size
        step_size_valid = val_gen.n//val_gen.batch_size

        model.fit_generator(train_gen, epochs=epochs,
                    steps_per_epoch=step_size_train, validation_data=val_gen,
                     validation_steps=step_size_valid, shuffle=True, verbose=2)

        return dataloader
