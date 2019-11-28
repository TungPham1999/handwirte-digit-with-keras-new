# save the final model to file
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping

# load train and test dataset


def load_dataset():
	# split ra thành train và val
	(trainX, trainY), (testX, testY) = mnist.load_data()
	trainX = trainX.reshape((trainX.shape[0], 28, 28))
	trainX, valX, trainY, valY = train_test_split(
	    trainX, trainY, test_size=0.25, random_state=42)

	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	valX = valX.reshape((valX.shape[0], 28, 28, 1))

	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	valY = to_categorical(valY)

	return trainX, trainY, testX, testY, valX, valY

# scale pixels


def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm


def normalize(list_of_subsets):
	# convert from integers to floats
	for i, _ in enumerate(list_of_subsets):
		list_of_subsets[i] = list_of_subsets[i].astype('float32')
		list_of_subsets[i] = list_of_subsets[i] / 255.0
	return list_of_subsets

# define cnn model


def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu',
	          kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu',
	          kernel_initializer='he_uniform'))
	model.add(Conv2D(64, (3, 3), activation='relu',
	          kernel_initializer='he_uniform'))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy',
	              metrics=['accuracy'])
	return model


# evaluate the deep model on the test dataset

# run the test harness for evaluating a model

def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY, valX, valY = load_dataset()

	# prepare pixel data
	[trainX, valX, testX] = normalize([trainX, valX, testX])

	# define model
	model = define_model()
	
	callbacks = [EarlyStopping(monitor='val_acc', mode='max', patience=3)]

	# fit model
	model.fit(trainX, trainY,
			epochs=1000,
			batch_size=16,
			verbose=2,
			validation_data=(valX, valY),
			callbacks=callbacks)

	# save model
	model.save('final_model.h5')

	# load model
	loaded_model=load_model('final_model.h5')

	# evaluate model on test dataset
	_, acc=loaded_model.evaluate(testX, testY, verbose=0)
	print('> %.3f' % (acc * 100.0))

# entry point, run the test harness
run_test_harness()


# make a prediction for a new image.
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

# load and prepare the image
def load_image(filename):
	# load the image
	img=load_img(filename, grayscale=True, target_size=(28, 28))
	# convert to array
	img=img_to_array(img)
	# reshape into a single sample with 1 channel
	img=img.reshape(1, 28, 28, 1)
	# prepare pixel data
	img=img.astype('float32')
	img=img / 255.0
	return img

# load an image and predict the class
def run_example():
	# load the image
	img=load_image('41.png')
	# load model
	model=load_model('final_model.h5')
	# predict the class
	digit=model.predict_classes(img)
	print(digit[0])

# entry point, run the example
run_example()
