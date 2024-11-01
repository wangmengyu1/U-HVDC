from keras.models import Model
from keras.layers import Input, core, Dropout, concatenate
from keras.layers.convolutional import Conv1D, MaxPooling1D, UpSampling1D
from keras.layers import LSTM, Dense, Flatten

# Define the Unet model with parameters for the number of classes, optimizer, input length, and number of channels
def Unet(nClasses, optimizer=None, input_length=1440, nChannels=1):  
    print("Number of classes nClasses received by the model:", nClasses)
    print("Current convolution kernel size: 64   Five-layer structure")
    
    inputs = Input((input_length, nChannels))  # Input shape with specified length and channels
    print("Input shape:", inputs.shape)
    
    # Encoder part
    conv1 = Conv1D(16, 64, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    print("Shape after first convolution (conv1):", conv1.shape)
    conv1 = Conv1D(16, 64, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    print("Shape after second convolution (conv1):", conv1.shape)
    pool1 = MaxPooling1D(pool_size=2)(conv1)
    print("Shape after first pooling (pool1):", pool1.shape)

    # Second layer
    conv2 = Conv1D(32, 64, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Dropout(0.1)(conv2)  # Original dropout rate was 0.2
    conv2 = Conv1D(32, 64, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling1D(pool_size=2)(conv2)

    # Third layer
    conv3 = Conv1D(64, 64, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Dropout(0.1)(conv3)  # New dropout layer
    conv3 = Conv1D(64, 64, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling1D(pool_size=2)(conv3)

    # Fourth layer
    conv4 = Conv1D(128, 64, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    print("Shape after fourth convolution (conv4):", conv4.shape)
    conv4 = Conv1D(128, 64, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    print("Shape after second convolution (conv4):", conv4.shape)
    pool4 = MaxPooling1D(pool_size=2)(conv4)
    print("Shape after fourth pooling (pool4):", pool4.shape)

    # Bottleneck
    conv5 = Conv1D(256, 64, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    print("Shape after fifth convolution (conv5):", conv5.shape)
    conv5 = Dropout(0.4)(conv5)  # Original dropout was 0.5
    print("Shape after dropout (conv5):", conv5.shape)
    conv5 = Conv1D(256, 64, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    print("Shape after second convolution (conv5):", conv5.shape)

    # Decoder part
    up1 = Conv1D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(conv5))
    print("Shape after first upsampling (up1):", up1.shape)
    merge1 = concatenate([up1, conv4], axis=-1)  # Concatenate skip connection
    print("Shape after first merge (merge1):", merge1.shape)
    conv6 = Conv1D(128, 64, activation='relu', padding='same', kernel_initializer='he_normal')(merge1)
    print("Shape after sixth convolution (conv6):", conv6.shape)
    conv6 = Conv1D(128, 64, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    print("Shape after second convolution (conv6):", conv6.shape)

    # Second upsampling
    up2 = Conv1D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(conv6))
    print("Shape after second upsampling (up2):", up2.shape)
    merge2 = concatenate([up2, conv3], axis=-1)  # Concatenate skip connection
    print("Shape after second merge (merge2):", merge2.shape)
    conv7 = Conv1D(64, 64, activation='relu', padding='same', kernel_initializer='he_normal')(merge2)
    print("Shape after seventh convolution (conv7):", conv7.shape)
    conv7 = Dropout(0.1)(conv7)  # New dropout layer
    print("Shape after dropout (conv7):", conv7.shape)
    conv7 = Conv1D(64, 64, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    print("Shape after second convolution (conv7):", conv7.shape)

    # Third upsampling
    up3 = Conv1D(32, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(conv7))
    print("Shape after third upsampling (up3):", up3.shape)
    merge3 = concatenate([up3, conv2], axis=-1)  # Concatenate skip connection
    print("Shape after third merge (merge3):", merge3.shape)
    conv8 = Conv1D(32, 64, activation='relu', padding='same', kernel_initializer='he_normal')(merge3)
    print("Shape after eighth convolution (conv8):", conv8.shape)
    conv8 = Dropout(0.1)(conv8)  # Original dropout was 0.2
    print("Shape after dropout (conv8):", conv8.shape)
    conv8 = Conv1D(32, 64, activation='relu', padding='same')(conv8)
    print("Shape after second convolution (conv8):", conv8.shape)

    # Fourth upsampling
    up4 = Conv1D(16, 2, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling1D(size=2)(conv8))
    print("Shape after fourth upsampling (up4):", up4.shape)
    merge4 = concatenate([up4, conv1], axis=-1)  # Concatenate skip connection
    print("Shape after fourth merge (merge4):", merge4.shape)
    conv9 = Conv1D(16, 64, activation='relu', padding='same', kernel_initializer='he_normal')(merge4)
    print("Shape after ninth convolution (conv9):", conv9.shape)
    conv9 = Conv1D(16, 64, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    print("Shape after second convolution (conv9):", conv9.shape)

    # Final output layer
    conv10 = Conv1D(nClasses, 1, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    print("Shape after final convolution (conv10):", conv10.shape)
    conv10 = core.Reshape((nClasses, input_length))(conv10)
    print("Shape after reshaping (conv10):", conv10.shape)
    conv10 = core.Permute((2, 1))(conv10)  # Permute dimensions for output
    print("Shape after permuting (conv10):", conv10.shape)

    conv11 = core.Activation('softmax')(conv10)  # Final activation layer

    # Create the model
    model = Model(inputs=inputs, outputs=conv11)
    
    # Compile the model if an optimizer is provided
    if optimizer is not None:
        model.compile(loss="categorical_crossentropy", optimizer=optimizer, metrics=['accuracy'])

    return model

if __name__ == '__main__':
    print("-------------2----------------")
    print('\nSummarizing the model:\n')
    model = Unet(2)  # Number of classes (2)
    model.summary()
    print('\nEnd of summary.\n')
