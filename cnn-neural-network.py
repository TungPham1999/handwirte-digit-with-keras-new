
# example of loading the mnist dataset
from keras.datasets import mnist
from matplotlib import pyplot
from keras.utils import np_utils
# load dataset
(trainX, trainy), (testX, testy) = mnist.load_data()
# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# plot first few images
for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# plot raw pixel data
	pyplot.imshow(trainX[i], cmap=pyplot.get_cmap('gray'))
# show the figure
#### pyplot.show()

# Convolutional neural networks
#API Keras hỗ trợ điều này bằng cách chỉ định đối số "verify_data" cho hàm model.fit () khi đào tạo mô hình, đến lượt nó, 
# sẽ trả về một đối tượng mô tả hiệu suất mô hình cho tổn thất và số liệu đã chọn trên mỗi train epoch .
# record model performance on a validation dataset during training
# history = model.fit(..., validation_data=(valX, valY))

#để ước tính được hiệu quả của model , chúng ta có thể sử dụng K-fold cross-validation


# example of k-fold cv for a neural net
# data = ...
# model = ...
# # prepare cross validation
# kfold = KFold(5, shuffle=True, random_state=1)
# # enumerate splits
# for train_ix, test_ix in kfold.split(data):
# 	...

# ======================================
# BASELINE MODEL

# baseline cnn model for mnist
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD

# load train and test dataset
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = mnist.load_data()
	# reshape dataset to have a single channel
	trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
	testX = testX.reshape((testX.shape[0], 28, 28, 1))
	# one hot encode target values
    # biến đổi số thành vector nhị phân chỉ gồm với 1 và 0 
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# scale pixels giới hạn là từ black to white(0 -> 255)
# chuyển đổi sang phạm vi từ [0,1] chia lại tỷ lệ lệ của ảnh chỉ còn là 0 và 1
def prep_pixels(train, test):
	# convert from integers to floats
	train_norm = train.astype('float32')
	test_norm = test.astype('float32')
	# normalize to range 0-1
	train_norm = train_norm / 255.0
	test_norm = test_norm / 255.0
	# return normalized images
	return train_norm, test_norm

# define cnn model
#có 2 phần chính cần nhớ
# 1: mặt trước khai thác tính năng bao gồm lớp chập và lớp gộp 
# 2: phía sau phân loại đưa ra dự đoán
# chúng ta sử dụng ReLU activation functuion và the "He" khởi tạo sơ đồ trọng lượng
#sử dụng stochastic gradient descent để tốt ưu learning rate từ 0.01 đến 0.9
def define_model():
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
	model.add(Dense(10, activation='softmax'))
	# compile model
	opt = SGD(lr=0.01, momentum=0.9)
	model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# evaluate a model using k-fold cross-validation
#đánh gái bằng five-dold crows-validation
def evaluate_model(model, dataX, dataY, n_folds=5):
	scores, histories = list(), list()
	# prepare cross validation
	kfold = KFold(n_folds, shuffle=True, random_state=1)
    # mỗi lần test ta sẽ shuffed (xáo trộn) dữ liệu test
	# enumerate splits
	for train_ix, test_ix in kfold.split(dataX):
		# select rows for train and test
		trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
		# fit model
		history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
		# evaluate model
		_, acc = model.evaluate(testX, testY, verbose=0)
		print('> %.3f' % (acc * 100.0))
		# stores scores
		scores.append(acc)
		histories.append(history)
	return scores, histories

# plot diagnostic learning curves
#hiển thị chuẩn đoán dựa trên lịch sử đào tạo được thu thập
def summarize_diagnostics(histories):
	for i in range(len(histories)):
		# plot loss
		pyplot.subplot(211)
		pyplot.title('Cross Entropy Loss')
		pyplot.plot(histories[i].history['loss'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
		# plot accuracy
		pyplot.subplot(212)
		pyplot.title('Classification Accuracy')
		pyplot.plot(histories[i].history['acc'], color='blue', label='train')
		pyplot.plot(histories[i].history['val_acc'], color='orange', label='test')
	pyplot.show()

# summarize model performance
#thực hiện điều này cho một danh sách điểm số nhất định được thu thập trong quá trình đánh giá mô hình.
def summarize_performance(scores):
	# print summary
	print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores)*100, std(scores)*100, len(scores)))
	# box and whisker plots of results
	pyplot.boxplot(scores)
	pyplot.show()

# run the test harness for evaluating a model
def run_test_harness():
	# load dataset
	trainX, trainY, testX, testY = load_dataset()
	# prepare pixel data
	trainX, testX = prep_pixels(trainX, testX)
	# define model
	model = define_model()
	# evaluate model
	scores, histories = evaluate_model(model, trainX, trainY)
	# learning curves
	summarize_diagnostics(histories)
	# summarize estimated performance
	summarize_performance(scores)

# entry point, run the test harness
run_test_harness()