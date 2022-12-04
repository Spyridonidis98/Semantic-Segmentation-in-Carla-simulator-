from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda, BatchNormalization, Dropout, Activation, Concatenate, MaxPool2D, Rescaling

def conv_block(input, num_filters, kernel_size):
    x = Conv2D(num_filters, kernel_size, padding="same")(input)
    x = BatchNormalization()(x)  
    x = Activation("relu")(x)

    x = Conv2D(num_filters, kernel_size, padding="same")(x)
    x = BatchNormalization()(x) 
    x = Activation("relu")(x)

    return x

#Encoder block: Conv block followed by maxpooling


def encoder_block(input, num_filters, kernel_size):
    x = conv_block(input, num_filters, kernel_size)
    p = MaxPool2D((2, 2))(x)
    return x, p   

#Decoder block
#skip features gets input from encoder for concatenation

def decoder_block(input, skip_features, num_filters, kernel_size):
    x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
    x = Concatenate()([x, skip_features])
    x = conv_block(x, num_filters, kernel_size)
    return x

#Build Unet using the blocks
def unet(n_classes, input_shape, kernel_size = (3,3)):
    scale  = 1/2
    
    inputs = Input(input_shape)
    s = Rescaling(1.0/255.0)(inputs)
    s1, p1 = encoder_block(s, 64*scale, kernel_size)
    s2, p2 = encoder_block(p1, 128*scale, kernel_size)
    s3, p3 = encoder_block(p2, 256*scale, kernel_size)
    s4, p4 = encoder_block(p3, 512*scale, kernel_size)

    b1 = conv_block(p4, 1024*scale, kernel_size) #Bridge

    d1 = decoder_block(b1, s4, 512*scale, kernel_size)
    d2 = decoder_block(d1, s3, 256*scale, kernel_size)
    d3 = decoder_block(d2, s2, 128*scale, kernel_size)
    d4 = decoder_block(d3, s1, 64*scale, kernel_size)

    outputs = Conv2D(n_classes, (1, 1), activation='softmax', dtype="float32")(d4)
    model = Model(inputs, outputs, name="U-Net")
    return model
