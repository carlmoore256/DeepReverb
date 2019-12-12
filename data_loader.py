from keras.preprocessing.image import ImageDataGenerator

class DataLoader:

    train_data = ''
    val_data = ''
    test_data = ''
    batch_size = 1
    img_width = 128
    img_height = 128
    datagen = ImageDataGenerator()

    def __init__(self, batch_size, img_width, img_height):
        self.image_width = img_width
        self.image_height = img_height
        self.train_data = './data/train/'
        self.val_data = './data/val/'
        self.test_data = './data/test/'
        self.batch_size = batch_size
        self.datagen = ImageDataGenerator(rescale=1./255, zoom_range=0, validation_split=0.3)

    def TrainGen(self):
        self.datagen = ImageDataGenerator(rescale=1./255, zoom_range=0, validation_split=0.3)
        train_gen = self.datagen.flow_from_directory(self.train_data,
            class_mode='categorical', batch_size=self.batch_size,
            target_size=(self.img_width, self.img_height), subset='training')

        x_train, y_train = train_gen.next()
        return train_gen

    def ValGen(self):
        self.datagen = ImageDataGenerator(rescale=1./255, zoom_range=0, validation_split=0.3)
        val_gen = self.datagen.flow_from_directory(self.val_data,
            class_mode='categorical', batch_size=self.batch_size,
            target_size=(self.img_width, self.img_height), subset='validation')
        return val_gen

    def TestGen(self):
        self.datagen = ImageDataGenerator(rescale=1./255, zoom_range=0, validation_split=0.3)
        test_gen = self.datagen.flow_from_directory(self.test_data,
            class_mode='categorical', batch_size=self.batch_size,
            target_size=(self.img_width, self.img_height), subset='training')
        return test_gen
