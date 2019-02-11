# Convolutional Neural Network
# to classify the image as cat or dog

# Part 1 - Building the CNN
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

#initialize the CNN
classifier = Sequential()

#step1 convolution + rectifier step
#start with 32 feature detector since using cpu
#input_shape will choose how to process out input for example 3 layers of R / G /B
# in this case since we using the tensor flow backend the order for input_shape is reverse
# and since using the CPU only the quality of the images was downgraded to (64,64,3)
classifier.add(Convolution2D(32 ,3 ,3, input_shape = (64,64,3), activation = 'relu'))

#step 2 MaxPooling method
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# adding the second layer of Conlolutional and maxPolling layers to increase the accuracy
classifier.add(Convolution2D(32 ,3 ,3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

#step 3 Flattening
classifier.add(Flatten())

#step4 creat the ANN part
# 128 dimentions / should be a power of 2 / not too high not too low
# rectifier activation function for the hidden layer (Fully connected layer)
# sigmoid function for the final layer and 1 dimention
classifier.add(Dense(output_dim = 128 , activation = 'relu'))
classifier.add(Dense(output_dim = 1 , activation = 'sigmoid'))

#compiling the CNN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'] )

#part 2 - fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
                                                    'dataset/training_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

test_set = test_datagen.flow_from_directory(
                                                    'dataset/test_set',
                                                    target_size=(64, 64),
                                                    batch_size=32,
                                                    class_mode='binary')

classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data=test_set,
        validation_steps=2000)
