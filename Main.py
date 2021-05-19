from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import plot_model
import matplotlib.pyplot as plt
from keras.models import load_model
import numpy as np

for abc in range(1,6):
    unit1 = 800 * abc
    unit2 = 400 * abc
    unit3 = 10 * abc
    (xTrain, labelsTrain), (xTest, labelsTest) = mnist.load_data()
    yTrain = to_categorical(labelsTrain, 10)
    if abc == 1:
        yTest = to_categorical(labelsTest, 10)

    xTrain = xTrain.astype('float32')
    xTest = xTest.astype('float32')
    xTrain /= 255
    xTest /= 255

    # ! Dense
    xTrain = xTrain.reshape(60000, 784)
    xTest = xTest.reshape(10000, 784)

    # ! Convolution
    # xTrain = xTrain.reshape(xTrain.shape[0], 28, 28, 1)
    # xTest = xTest.reshape(xTest.shape[0], 28, 28, 1)

    net = Sequential()
    net.add(Dense(unit1, activation='relu', input_shape=(784,)))
    net.add(Dense(unit2, activation='relu'))
    net.add(Dense(unit3, activation='softmax'))

    net.summary()
    plot_model(net, to_file='netStruct.png', show_shapes=True)

    net.compile(loss='categorical_crossentropy', optimizer='adam')
    history = net.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=1, batch_size=256)

    plt.figure()
    plt.plot(history.history['loss'], label='training_loss')
    plt.plot(history.history['val_loss'], label='validation_loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    #plt.show()

    #net.save("initialNN.h5")

    outputs = net.predict(xTest)
    labelsPredicted = np.argmax(outputs, axis=1)
    misclassified = sum(labelsPredicted != labelsTest)
    print('Percentage misclassified = ', 100*misclassified/labelsTest.size)

    # plt.figure(figsize=(8, 2))
    # for i in range(0, 8):
    #     ax = plt.subplot(2, 8, i+1)
    #     plt.imshow(xTest[i,:].reshape(28,28), cmap=plt.get_cmap('gray_r'))
    #     plt.title(labelsTest[i])
    #     ax.get_xaxis().set_visible(False)
    #     ax.get_yaxis().set_visible(False)

    # for i in range(0, 8):
    #     output = net.predict(xTest[i,:].reshape(1, 784))
    #     output = output[0,0:]
    #     plt.subplot(2, 8, 8+i+1)
    #     plt.bar(np.arange(10.), output)
    #     plt.title(np.argmax(output))

    # plt.show()
