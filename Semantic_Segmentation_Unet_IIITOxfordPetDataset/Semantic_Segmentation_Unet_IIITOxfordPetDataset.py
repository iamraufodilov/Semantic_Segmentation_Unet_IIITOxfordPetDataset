# import libraries
import os
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, MaxPooling2D
from tensorflow.keras import Model
# load dataset
IMG_PATH = 'G:/rauf/STEPBYSTEP/Data/oxford_pets/images/'
ANNOTATION_PATH = 'G:/rauf/STEPBYSTEP/Data/oxford_pets/annotations/trimaps/'

input_img_paths = sorted(
    [
        os.path.join(IMG_PATH, fname)
        for fname in os.listdir(IMG_PATH)
        if fname.endswith(".jpg")
    ]
)
annotation_img_paths = sorted(
    [
        os.path.join(ANNOTATION_PATH, fname)
        for fname in os.listdir(ANNOTATION_PATH)
        if fname.endswith(".png") and not fname.startswith(".")
    ]
)

print(len(input_img_paths), len(annotation_img_paths))


IMG_SHAPE = 128
AUTO = tf.data.experimental.AUTOTUNE
BATCH_SIZE = 32

def scale_down(image, mask):
  # apply scaling to image and mask
  image = tf.cast(image, tf.float32) / 255.0
  mask -= 1
  return image, mask

def load_and_preprocess(img_filepath, mask_filepath):
   # load the image and resize it
    img = tf.io.read_file(img_filepath)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_SHAPE, IMG_SHAPE])

    mask = tf.io.read_file(mask_filepath)
    mask = tf.io.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, [IMG_SHAPE, IMG_SHAPE])

    img, mask = scale_down(img, mask)

    return img, mask

# shuffle the paths and prepare train-test split
input_img_paths  = tf.random.shuffle(input_img_paths, seed=42)
annotation_img_paths = tf.random.shuffle(annotation_img_paths, seed=42)
input_img_paths_train, annotation_img_paths_train = input_img_paths[: -1000], annotation_img_paths[: -1000]
input_img_paths_test, annotation_img_paths_test = input_img_paths[-1000:], annotation_img_paths[-1000:]

trainloader = tf.data.Dataset.from_tensor_slices((input_img_paths_train, annotation_img_paths_train))
testloader = tf.data.Dataset.from_tensor_slices((input_img_paths_test, annotation_img_paths_test))

trainloader = (
    trainloader
    .shuffle(1024)
    .map(load_and_preprocess, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

testloader = (
    testloader
    .map(load_and_preprocess, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

# create the model

class SegmentationModel:
    # build unet architecture for image semantic segmentation task
    def prepare_model(self, OUTPUT_CHANNEL, input_size=(IMG_SHAPE,IMG_SHAPE,3)):
        inputs = tf.keras.layers.Input(input_size)

        # Encoder 
        conv1, pool1 = self.__ConvBlock(32, (3,3), (2,2), 'relu', 'same', inputs) 
        conv2, pool2 = self.__ConvBlock(64, (3,3), (2,2), 'relu', 'same', pool1)
        conv3, pool3 = self.__ConvBlock(128, (3,3), (2,2), 'relu', 'same', pool2) 
        conv4, pool4 = self.__ConvBlock(256, (3,3), (2,2), 'relu', 'same', pool3) 
    
        # Decoder
        conv5, up6 = self.__UpConvBlock(512, 256, (3,3), (2,2), (2,2), 'relu', 'same', pool4, conv4)
        conv6, up7 = self.__UpConvBlock(256, 128, (3,3), (2,2), (2,2), 'relu', 'same', up6, conv3)
        conv7, up8 = self.__UpConvBlock(128, 64, (3,3), (2,2), (2,2), 'relu', 'same', up7, conv2)
        conv8, up9 = self.__UpConvBlock(64, 32, (3,3), (2,2), (2,2), 'relu', 'same', up8, conv1)
    
        conv9 = self.__ConvBlock(32, (3,3), (2,2), 'relu', 'same', up9, False)

        outputs = Conv2D(OUTPUT_CHANNEL, (3, 3), activation='softmax', padding='same')(conv9)

        return Model(inputs=[inputs], outputs=[outputs]) 

    def __ConvBlock(self, filters, kernel_size, pool_size, activation, padding, connecting_layer, pool_layer=True):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(connecting_layer)
        conv = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv)
        if pool_layer:
            pool = MaxPooling2D(pool_size)(conv)
            return conv, pool
        else:
            return conv

    def __UpConvBlock(self, filters, up_filters, kernel_size, up_kernel, up_stride, activation, padding, connecting_layer, shared_layer):
        conv = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(connecting_layer)
        conv = Conv2D(filters=filters, kernel_size=kernel_size, activation=activation, padding=padding)(conv)
        up = Conv2DTranspose(filters=up_filters, kernel_size=up_kernel, strides=up_stride, padding=padding)(conv)
        up = tf.keras.layers.concatenate([up, shared_layer], axis=3)

        return conv, up



OUTPUT_CHANNEL = 3

model = SegmentationModel().prepare_model(OUTPUT_CHANNEL)
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

model.summary()

# Callback - Interactive Visualization of Predictions
segmentation_classes = ['pet', 'pet_outline', 'background']

def labels():
    l={}
    for i , label in enumerate(segmentation_classes):
        l[i] = label
    return l

def wandb_mask(bg_img, pred_mask, true_mask):
    return wandb.Image(bg_img, masks={
        "prediction":{
            "mask_data": pred_mask,
            "class_labels": labels()
            },
        "ground_truth":{
            "mask_data": true_mask,
            "class_labels": labels()
            }
        }
                       )

class SemanticLogger(tf.keras.callbacks.Callback):
    def __init__(self):
        super(SemanticLogger, self).__init__()
        self.val_images, self.val_masks = next(iter(testloader))

    def on_epoch_end(self, logs, epoch):
        pred_masks = self.model.predict(self.val_images)
        pred_masks = np.argmax(pred_masks, axis=-1)
        # pred_masks = np.expand_dims(pred_masks, axis=-1)

        val_images = tf.image.convert_image_dtype(self.val_images, tf.uint8)
        val_masks = tf.image.convert_image_dtype(self.val_masks, tf.uint8)
        val_masks = tf.squeeze(val_masks, axis=-1)
        
        pred_masks = tf.image.convert_image_dtype(pred_masks, tf.uint8)

        mask_list = []
        for i in range(len(self.val_images)):
          mask_list.append(wandb_mask(val_images[i].numpy(), 
                                      pred_masks[i].numpy(), 
                                      val_masks[i].numpy()))

        wandb.log({"predictions" : mask_list})


model.fit(trainloader, batch_size=BATCH_SIZE, epochs=10, verbose='auto')


# this model is unet architecture for task of Semantic segmentation
# the dataset used in this experiment IIIT Oxford Pet Dataset
