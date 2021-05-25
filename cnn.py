from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
# Initialising the CNN
classifier = Sequential()
# Step 1 - Convolution1
classifier.add(Conv2D(32,(3,3), input_shape = (64, 64, 3), activation = 'relu'))
# Step 2 - Pooling1
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 3 - Convolution2
classifier.add(Conv2D(32, (3,3), activation = 'relu'))
# Step 4 - Pooling2
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Step 5 - Flattening
classifier.add(Flatten())
# Step 4 - Full connection Network
classifier.add(Dense(units = 128, activation = 'relu'))
classifier.add(Dense(units = 3, activation = 'softmax'))
# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images
from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('G:\Data',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('G:\Data',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'categorical')

model = classifier.fit_generator(training_set,
                         steps_per_epoch = 1000,
                         epochs = 1,
                         validation_data = test_set,)


classifier.save("model_new.h5")
print("Saved model to disk")

print(training_set.class_indices)

