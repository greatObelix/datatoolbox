# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 18:28:08 2018

sample neural network with Kera
"""

# MLP: Multi Layer Perceptron neural net
def MLP ():
    import numpy as np
    from keras.models import Sequential
    from keras.layers.core import Dense, Activation
    
    # X has shape (num_rows, num_cols), where the training data are stored
    # as row vectors
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    
    # y must have an output vector for each input vector
    y = np.array([[0], [0], [0], [1]], dtype=np.float32)
    
    # One-hot encoding the output
    # if using loss="categorical_crossentropy"
    y = np_utils.to_categorical(y)
    
    # Create the Sequential model
    model = Sequential()
    
    # 1st Layer - Add an input layer of 32 nodes with the same input shape as
    # the training samples in X
    model.add(Dense(32, input_dim=X.shape[1]))
    
    # Add a softmax activation layer
    model.add(Activation('softmax'))
    
    # 2nd Layer - Add a fully connected output layer
    model.add(Dense(1))
    
    # Add a sigmoid activation layer
    model.add(Activation('sigmoid'))
    
    #Compiling the Keras model calls the backend (tensorflow, theano, etc.) and binds the optimizer, loss function
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics = ["accuracy"])
    
    # see results
    model.summary()
    
    #specifies the number of training epochs and the message level (how much information we want displayed on the screen during training).
    history = model.fit(X, y, nb_epoch=1000, verbose=0)
    
    # Scoring the model
    score = xor.evaluate(X, y)
    print("\nAccuracy: ", score[-1])
    
    # Checking the predictions
    print("\nPredictions:")
    print(xor.predict_proba(X))

# MLP version 2    
def MLPv2():
    import keras
    from keras.datasets import cifar10
    
    # load the pre-shuffled train and test data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()        

    # Visualize the data
    import numpy as np
    import matplotlib.pyplot as plt        
    
    fig = plt.figure(figsize=(20,5))
    for i in range(36):
        ax = fig.add_subplot(3, 12, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(x_train[i]))

    # rescale [0,255] --> [0,1]. Rescale the Images by Dividing Every Pixel in Every Image by 255
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    # break into test/train/validation
    from keras.utils import np_utils
    
    # one-hot encode the labels
    num_classes = len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    # break training set into training and validation sets
    (x_train, x_valid) = x_train[5000:], x_train[:5000]
    (y_train, y_valid) = y_train[5000:], y_train[:5000]
    
    # print shape of training set
    print('x_train shape:', x_train.shape)
    
    # print number of training, validation, and test images
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print(x_valid.shape[0], 'validation samples')

    # Define Model architecture
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten
    
    # define the model
    model = Sequential()
    model.add(Flatten(input_shape = x_train.shape[1:]))
    model.add(Dense(1000, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))

    model.summary()
    
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    #train the model
    from keras.callbacks import ModelCheckpoint   
    
    # train the model
    checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, 
                                   save_best_only=True)
    hist = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_valid, y_valid), callbacks=[checkpointer], verbose=2, shuffle=True)


    # load the weights that yielded the best validation accuracy
    model.load_weights('model.weights.best.hdf5')

    # evaluate and print test accuracy
    score = model.evaluate(x_test, y_test, verbose=0)
    print('\n', 'Test accuracy:', score[1])

    # get predictions on the test set
    y_hat = model.predict(x_test)
    
# CNN: convoluted neural net
def CNN():
 
    import keras
    from keras.datasets import cifar10
    
    # load the pre-shuffled train and test data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()        

    # Visualize the data
    import numpy as np
    import matplotlib.pyplot as plt        
    
    fig = plt.figure(figsize=(20,5))
    for i in range(36):
        ax = fig.add_subplot(3, 12, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(x_train[i]))

    # rescale [0,255] --> [0,1]. Rescale the Images by Dividing Every Pixel in Every Image by 255
    x_train = x_train.astype('float32')/255
    x_test = x_test.astype('float32')/255

    # break into test/train/validation
    from keras.utils import np_utils
    
    # one-hot encode the labels
    num_classes = len(np.unique(y_train))
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    
    # break training set into training and validation sets
    (x_train, x_valid) = x_train[5000:], x_train[:5000]
    (y_train, y_valid) = y_train[5000:], y_train[:5000]
    
    # print shape of training set
    print('x_train shape:', x_train.shape)
    
    # print number of training, validation, and test images
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')
    print(x_valid.shape[0], 'validation samples')

    # create & configure augemented image generator
    from keras.preprocessing.image import ImageDataGenerator
    
    # create and configure augmented image generator
    datagen_train = ImageDataGenerator(
        width_shift_range=0.1,  # randomly shift images horizontally (10% of total width)
        height_shift_range=0.1,  # randomly shift images vertically (10% of total height)
        horizontal_flip=True) # randomly flip images horizontally
    
    # fit augmented image generator on data
    datagen_train.fit(x_train)
    
    #Visualize original and Augmented Images
    import matplotlib.pyplot as plt
    
    # take subset of training data
    x_train_subset = x_train[:12]
    
    # visualize subset of training data
    fig = plt.figure(figsize=(20,2))
    for i in range(0, len(x_train_subset)):
        ax = fig.add_subplot(1, 12, i+1)
        ax.imshow(x_train_subset[i])
    fig.suptitle('Subset of Original Training Images', fontsize=20)
    plt.show()
    
    # visualize augmented images
    fig = plt.figure(figsize=(20,2))
    for x_batch in datagen_train.flow(x_train_subset, batch_size=12):
        for i in range(0, 12):
            ax = fig.add_subplot(1, 12, i+1)
            ax.imshow(x_batch[i])
        fig.suptitle('Augmented Images', fontsize=20)
        plt.show()
        break;

    # Define Model architecture
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
   
    model = Sequential()
    #These first six layers are designed to take the input array of image pixels and convert it to an array 
    #where all of the spatial information has been squeezed out, and only information encoding the content 
    #of the image remains. The array is then flattened to a vector in the seventh layer of the CNN. 
    #It is followed by two dense layers designed to further elucidate the content of the image. 
    #The final layer has one entry for each object class in the dataset, and has a softmax 
    #activation function, so that it returns probabilities
    
    #create a convolutional layer        
    #filters - The number of filters.
    #kernel_size - Number specifying both the height and width of the (square) convolution window.
    #strides - The stride of the convolution. If you don't specify anything, strides is set to 1.
    #padding - One of 'valid' or 'same'. If you don't specify anything, padding is set to 'valid'.
    #activation - Typically 'relu'. If you don't specify anything, no activation is applied. You are strongly encouraged to add a ReLU activation function to every convolutional layer in your networks.
    #When using your convolutional layer as the first layer (appearing after the input layer) in a model, you must provide an additional input_shape argument. Not otherwise
    #input_shape - Tuple specifying the height, width, and depth (in that order) of the input.
    model.add( \
        Conv2D(filters=16, kernel_size=(2,2), strides=2, padding='same', activation='relu', input_shape=(32,32,3))  \
        )
    
    #create a pooling layer
    # pool_size - Number specifying the height and width of the pooling window.
    # strides - The vertical and horizontal stride. If you don't specify anything, strides will default to pool_size.
    # padding - one of  default = 'valid' or 'same'
    #Say the convolutional layer has size (100, 100, 15), and I'd like the max pooling layer to have size (50, 50, 15).
    # I can do this by using a 2x2 window in my max pooling layer, with a stride of 2
    model.add(MaxPooling2D(pool_size=2, strides=2, input_shape=(100, 100, 15)))
    
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(10, activation='softmax'))

    model.summary()
    
    # compile the model
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    #train the model
    from keras.callbacks import ModelCheckpoint   

    batch_size = 32
    epochs = 100

    checkpointer = ModelCheckpoint(filepath='model.weights.best.hdf5', verbose=1, 
                                   save_best_only=True)
    #if without image augmentation use fit()
    #model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_valid, y_valid), callbacks=[checkpointer], verbose=2, shuffle=True)
    # if image augmentation use fit_generator()
    model.fit_generator(datagen_train.flow(x_train, y_train, batch_size=batch_size),
                    steps_per_epoch=x_train.shape[0] // batch_size,
                    epochs=epochs, verbose=2, callbacks=[checkpointer],
                    validation_data=(x_valid, y_valid),
                    validation_steps=x_valid.shape[0] // batch_size)

    # load the weights that yielded the best validation accuracy
    model.load_weights('model.weights.best.hdf5')

    # evaluate and print test accuracy
    score = model.evaluate(x_test, y_test, verbose=0)
    print('\n', 'Test accuracy:', score[1])
            
    # get predictions on the test set
    y_hat = model.predict(x_test)
    
    # define text labels (source: https://www.cs.toronto.edu/~kriz/cifar.html)
    cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    
    # plot a random sample of test images, their predicted labels, and ground truth
    fig = plt.figure(figsize=(20, 8))
    for i, idx in enumerate(np.random.choice(x_test.shape[0], size=32, replace=False)):
        ax = fig.add_subplot(4, 8, i + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(x_test[idx]))
        pred_idx = np.argmax(y_hat[idx])
        true_idx = np.argmax(y_test[idx])
        ax.set_title("{} ({})".format(cifar10_labels[pred_idx], cifar10_labels[true_idx]), color=("green" if pred_idx == true_idx else "red"))
                     
