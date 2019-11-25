# Plot ad hoc mnist instances
from keras.datasets import mnist
import matplotlib.pyplot as plt
# load (downloaded if needed) the MNIST dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# download and load data
# plot 4 images as gray scale
plt.subplot(221)
plt.imshow(X_train[0], cmap=plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(X_train[1], cmap=plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(X_train[2], cmap=plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(X_train[3], cmap=plt.get_cmap('gray'))
# show the plot
plt.show()

#=================================================================
##BASELINE MODEL WITH MULTI-LAYER PERCEPTIONS
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

#load the MNIST dataset using keras helper function
(X_train, y_train), (X_test, y_test) = mnist.load_data()

#  tập dữ liệu được cấu trúc như 1 array 3-dimensional , chiều cao và rộng của ảnh , chúng ta cần giảm chiều ảnh thành 1 vector
#  của pixels (28*28) 

# flatten 28*28 images to a 784 vector for each image
#using reshape() function on numpy array . chúng ta có thể giảm dung lượng của ảnh xuông mức 32bit
num_pixels = X_train.shape[1] * X_train.shape[2]
X_train = X_train.reshape((X_train.shape[0], num_pixels)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], num_pixels)).astype('float32')

# giá trị của pixels xám tỉ lệ từ 0 đén 255. chúng luôn là ý tưởng tốt dể thực hiện 1 số tỉ lệ giá trị đầu vào.
# Vì chúng ta có thể nhanh chóng bình thường hóa các giá trị pixel
# thành phạm vi 0 và 1 bằng cách chia mỗi giá trị cho tối đa 255.
# normalize inputs from 0-255 to 0-1

X_train = X_train / 255
X_test = X_test / 255

#finally thì output từ 0 -> 9. Đây là một vấn đề phân loại nhiều lớp. biến đổi vectơ của các 
# số nguyên lớp thành một ma trận nhị phân.

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

#xử lí để có số điểm train tốt hơn

...
# define baseline model
def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(num_pixels, input_dim=num_pixels, kernel_initializer='normal', activation='relu'))
	model.add(Dense(num_classes, kernel_initializer='normal', activation='softmax'))
    #hàm "softmax" được sử dụng trên lớp đầu để biến đầu ra thành xắc xuất
	# Compile model
    #Logarithmic loss được sử dụng là hàm loss function (categorical_crossentropy) và thuật toán "ADAM" 
    # thuật toán giảm độ dốc được sử dụng để tìm hiểu các trọng số.
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# build the model
model = baseline_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200, verbose=2)
# verbose : mỗi dòng 1 epoch
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))