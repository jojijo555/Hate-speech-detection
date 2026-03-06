import numpy as np
from pyforest import tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import keras
from keras import losses
from keras.layers import Input, MaxPooling2D, Reshape, Dropout, Concatenate, SimpleRNN
from keras.optimizers import Adam
from keras.utils import to_categorical, plot_model
from keras.layers import LeakyReLU
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Flatten
from keras.models import Sequential
from keras.layers import LSTM, Dense
from termcolor import cprint
from keras import layers, models
from keras.utils import to_categorical


def eval_matrix(Y_test, pred):
    cm = confusion_matrix(Y_test, pred)
    # cn = np.sum(cm, axis=0)

    TP = cm[0][0]
    TN = cm[1][1]
    FP = cm[1][0]
    FN = cm[0][1]

    Total = TP + TN + FP + FN
    Prevalence = (TP + FN) / Total
    # Accuracy
    ACC = (TP + TN) / Total
    # Sensitivity (SEN)/recall
    REC = TP / (TP + FN)
    # False negative rate
    FNR = FN / (TP + FN)
    # Specificity (SPC)
    SPC = TN / (TN + FP)
    # False positive rate
    FPR = FP / (TN + FP)
    # Matthews correlation coefficient (MCC)
    sqr = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    MCC = ((TP * TN) - (FP * FN)) / sqr

    # Positive and Negative Prediction Value(PPV)/Precision
    PRE = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    F1_score = 2 * TP / ((2 * TP) + FP + FN)
    PPV = TP / (TP + FP)
    TPR = TP / (TP + FN)
    return ACC, PPV, NPV


class model:
    def __init__(self, x_train, x_test, y_train, y_test, epochs):
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.epochs = epochs


    def CNN_LSTM(self):
        # Notify that the function is running
        cprint("CNN_LSTM is running", on_color='on_grey')
        # Convert labels to categorical format
        y_train = to_categorical(self.y_train)

        # Reshape the data for Conv2D layer (adding 1x1 channel for each pixel)
        x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1, 1)
        x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1, 1)

        # Define the input layer for the CNN-LSTM model
        inputlayer = Input((x_train.shape[1], x_train.shape[2], x_train.shape[3]))

        # CNN part: Conv2D and MaxPooling2D layers
        x1 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(inputlayer)
        x1 = MaxPooling2D((2, 2), padding='same')(x1)
        x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
        x1 = LeakyReLU(alpha=0.1)(x1)
        x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)
        x1 = Dropout(0.5)(x1)
        x1 = Flatten()(x1)

        # LSTM part: Reshape and pass through LSTM layers
        reshapelayer = Reshape((x_train.shape[1], x_train.shape[2] * x_train.shape[3]))(inputlayer)
        x2 = LSTM(232, activation='relu', return_sequences=True)(reshapelayer)
        x2 = LSTM(122, activation='relu', return_sequences=True)(x2)
        x2 = LSTM(182, activation='relu', return_sequences=True)(x2)
        x2 = LSTM(242, activation='relu', return_sequences=False)(x2)
        x2 = Dropout(0.5)(x2)
        x2 = Dense(150, activation="relu")(x2)
        x2 = Flatten()(x2)

        # Combine the CNN and LSTM outputs
        x = Concatenate()([x1, x2])

        # Fully connected layers
        x = Dense(100, activation='relu')(x)
        x = Dense(128, activation='relu')(x)

        # Output layer: softmax for classification
        outputlayer = Dense(y_train.shape[1], activation='softmax')(x)

        # Define and compile the model
        model = Model(inputs=inputlayer, outputs=outputlayer)
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])
        # Check the model summary to see the structure of the model
        model.summary()
        x_train = tf.convert_to_tensor(x_train, dtype=float)
        # Train the model
        model.fit(x_train, y_train, validation_split=0.2)

        # Make predictions on the test set
        pred1 = np.argmax(model.predict(x_test), axis=-1)

        return pred1

    def Capsule_Network(self):
        cprint("Capsule_Network is running", on_color='on_grey')
        model = Sequential()
        epochs = 2  # run
        Y_train1 = to_categorical(self.y_train)
        x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1)
        x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1)

        model.add(keras.layers.Bidirectional(keras.layers.LSTM(64, return_sequences=True)))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(32)))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.Dense(Y_train1.shape[1]))
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
        # X_train1 = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2]).astype(np.float64)
        # X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2]).astype(np.float64)
        model.fit(x_train, Y_train1, epochs=self.epochs, validation_split=0.2 )
        model.summary()
        pred6 = np.argmax(model.predict(x_test), axis=1)
        return pred6


    def cnn(self):
        cprint(f"cnn is Running", 'cyan', on_color='on_grey')
        x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1, 1)
        x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1, 1)
        # Define the CNN model
        y_train = to_categorical(self.y_train)
        input_layer = Input((x_train.shape[1], x_train.shape[2], x_train.shape[3]))
        x1 = Conv2D(32, kernel_size=(1, 1), activation='relu')(input_layer)
        x1 = MaxPooling2D((1, 1))(x1)
        x1 = LeakyReLU(alpha=0.1)(x1)
        x1 = Conv2D(64, kernel_size=(1, 1), activation='relu')(x1)
        x1 = MaxPooling2D((1, 1))(x1)
        x1 = Conv2D(64, kernel_size=(1, 1), activation='relu')(x1)
        x1 = MaxPooling2D((1, 1))(x1)
        x1 = LeakyReLU(alpha=0.1)(x1)
        x1 = Conv2D(8, (1, 1), activation='relu')(x1)
        x1 = MaxPooling2D(pool_size=(1, 1))(x1)

        x1 = Flatten()(x1)
        # x1 = Dense(36, activation='relu')(x1)
        x1 = Dense(16, activation='relu')(x1)
        x1 = Dense(8, activation='relu')(x1)
        # x1 = Dense(200, activation='relu')(x1)
        # x1 = Dense(100, activation='relu')(x1)
        output = Dense(y_train.shape[1], activation='softmax')(x1)
        model = Model(inputs=input_layer, outputs=output)
        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        # Train the model
        plot_model(model, to_file='CNN.png', show_shapes=True, show_layer_names=True, dpi=800)
        model.fit(self.x_train, self.y_train, epochs=2, batch_size=32, validation_split=0.2)
        # Predict on test data
        pred6 = model.predict(self.x_test)
        return pred6


    def MLHS_CGCapNet(self):
        # Print a message indicating that RNN model is running
        # One-hot encode the labels
        y_train = to_categorical(self.y_train)
        # Ensure input data is 3D for RNN (samples, time steps, features)
        x_train = self.x_train.reshape(self.x_train.shape[0], self.x_train.shape[1], 1)
        x_test = self.x_test.reshape(self.x_test.shape[0], self.x_test.shape[1], 1)
        # Create Sequential model
        model = Sequential()
        # Recurrent Neural Network using SimpleRNN
        model.add(SimpleRNN(10, input_shape=(x_train.shape[1], 1)))
        # Dropout layer
        model.add(Dropout(0.5))
        # Output layer
        model.add(Dense(y_train.shape[1], activation='softmax'))  # Number of classes
        # Compile the model
        model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # Fit the model for training data
        model.fit(self.x_train, self.y_train, epochs=2, batch_size=8)
        # Predict the model from test data
        pred = model.predict(self.x_test)
        pred = np.argmax(pred, axis=1)
        return pred


    def neural_netowrk(self):
        y_train = to_categorical(self.y_train)
        y_test = to_categorical(self.y_test)

        # Build a deeper neural network model
        model = models.Sequential([
            layers.Dense(512, activation='relu', input_shape=(784,)),
            layers.Dropout(0.3),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(10, activation='softmax')
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        model.fit(self.x_train, self.y_train, epochs=10, batch_size=64, validation_split=0.2)

        pred = model.predict(self.x_test)
        return pred


def TP_COMPARTIVE(FEAT, LAB, db):
    n = 0
    ACC, PPV, NPV = [], [], []
    T = [0.4, 0.5, 0.6, 0.7, 0.8]

    for i in range(len(T)):  # Training percentage
        # split the feat and lab
        x_train, x_test, y_train, y_test = train_test_split(FEAT, LAB, train_size=0.2)

        classfier = model(x_train, x_test, y_train, y_test, 10)
        # # # compartive
        mod_1 = classfier.CNN_LSTM()
        mod_2 = classfier.Capsule_Network()
        mod_3 = classfier.cnn()
        mod_4 = classfier.MLHS_CGCapNet()
        NEural_NeTwoRk = classfier.neural_netowrk()

        mod_5 = NEural_NeTwoRk(x_train, x_test, y_train, y_test, 1, 2, db)
        mod_6 = NEural_NeTwoRk(x_train, x_test, y_train, y_test, 10, 1, db)
        mod_7 = NEural_NeTwoRk(x_train, x_test, y_train, y_test, 10, 2, db)
        mod_8 = NEural_NeTwoRk(x_train, x_test, y_train, y_test, 10, 3, db)

        # #performance
        mod_9 = NEural_NeTwoRk(x_train, x_test, y_train, y_test, 1, 3, db)
        mod_10 = NEural_NeTwoRk(x_train, x_test, y_train, y_test, 10, 3, db)
        mod_11 = NEural_NeTwoRk(x_train, x_test, y_train, y_test, 10, 3, db)
        mod_12 = NEural_NeTwoRk(x_train, x_test, y_train, y_test, 10, 3, db)

        metrics1 = eval_matrix(mod_1, y_test)
        metrics2 = eval_matrix(mod_2, y_test)
        metrics3 = eval_matrix(mod_3, y_test)
        metrics4 = eval_matrix(mod_4, y_test)
        metrics5 = eval_matrix(mod_5, y_test)
        metrics6 = eval_matrix(mod_6, y_test)
        metrics7 = eval_matrix(mod_7, y_test)
        metrics8 = eval_matrix( mod_8, y_test)
        metrics9 = eval_matrix( mod_9, y_test)
        metrics10 = eval_matrix(mod_10, y_test)
        metrics11 = eval_matrix(mod_11, y_test)
        metrics12 = eval_matrix(mod_12, y_test)


        ACC.append([metrics1[0], metrics2[0], metrics3[0], metrics4[0], metrics5[0], metrics6[0], metrics7[0], metrics8[0], metrics9[0], metrics10[0], metrics11[0], metrics12[0]])

        PPV.append([metrics1[1], metrics2[1], metrics3[1], metrics4[1], metrics5[1], metrics6[1], metrics7[1], metrics8[1],  metrics9[1], metrics10[1],  metrics11[1],   metrics12[1]])

        NPV.append([metrics1[2], metrics2[2], metrics3[2], metrics4[2], metrics5[2], metrics6[2], metrics7[2], metrics8[2], metrics9[2], metrics10[2], metrics11[2],   metrics12[2]])

        MCC.append([metrics1[3], metrics2[3], metrics3[3], metrics4[3], metrics5[3], metrics6[3], metrics7[3], metrics8[3], metrics9[3], metrics10[3], metrics11[3],   metrics12[3]])

        print(('\033[46m' + '\033[30m' + "________________________Save Metrics__________________________________" + '\x1b[0m'))

        if db == 1:

            ## Save metrics
            np.save(f"Analysis2/TP1/ACC_1.npy", np.array(ACC))
            np.save(f"Analysis2/TP1/PPV_1.npy", np.array(PPV))
            np.save(f"Analysis2/TP1/NPV_1.npy", np.array(NPV))
            np.save(f"Analysis2/TP1/MCC_1.npy", np.array(MCC))
        else:

            np.save(f"Analysis2/TP1/ACC_2.npy", np.array(ACC))
            np.save(f"Analysis2/TP1/PPV_2.npy", np.array(PPV))
            np.save(f"Analysis2/TP1/NPV.npy", np.array(NPV))
            np.save(f"Analysis2/TP1/MCC_2.npy", np.array(MCC))
