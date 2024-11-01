# Import necessary libraries and modules
from Unet import Unet
import LoadBatches1D
import keras
from tensorflow import keras
from keras import optimizers
import warnings
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import time

# Disable TensorFlow v2 behavior
tf.compat.v1.disable_v2_behavior()

# Configure TensorFlow session for resource allocation
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

warnings.filterwarnings("ignore")
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"  # Allow dynamic memory allocation on GPU
tf.compat.v1.GPUOptions.per_process_gpu_memory_fraction = 0.8  # Limit GPU usage to 80%
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Ensure correct GPU ID matching
os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # Use GPU 0
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
tf.compat.v1.keras.backend.set_session(sess)

# Experiment ID
exp_id = '02FE_5'  # Modify this experiment ID

def lr_schedule(epoch):
    # Learning rate decay scheme during training
    lr = 0.00005  # Initial learning rate
    print('Learning rate: ', lr)
    return lr

# Path settings for training and validation data
train_sigs_path = './data/newsamplemark811/1418_train_geo/'
train_segs_path = './data/newsamplemark811/1418_train_label/'
val_sigs_path = './data/newsamplemark811/1418_val_geo/'
val_segs_path = './data/newsamplemark811/1418_val_label/'  # Modify path

# Directory for saving results
SAVE_DIR = './results' + '/{}'.format(exp_id)
if not os.path.exists(os.path.join(SAVE_DIR)):
    os.mkdir(os.path.join(SAVE_DIR))
print("SAVE_DIR", SAVE_DIR)

train_batch_size = 64  # Batch size for training
n_classes = 2  # Number of classes
input_length = 1440  # Length of input sequence
optimizer_name = optimizers.Adam(lr_schedule(0))  # Optimizer setup
PATIENCE = 20  # Number of epochs with no improvement before stopping

val_batch_size = 2  # Batch size for validation

lr_scheduler = keras.callbacks.LearningRateScheduler(lr_schedule)

# Initialize the U-Net model
model = Unet(n_classes, input_length=input_length)

# Compile the model with loss function and optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer_name,
              metrics=['accuracy'])

model.summary()  # Print model summary

print("e-code={}".format(exp_id))
print("PATIENCE={}acc".format(PATIENCE))

output_length = 86400  # Length of output sequence

# Create data generators for training and validation
G = LoadBatches1D.SigSegmentationGenerator(train_sigs_path, train_segs_path, train_batch_size, n_classes, output_length)
G2 = LoadBatches1D.SigSegmentationGenerator(val_sigs_path, val_segs_path, val_batch_size, n_classes, output_length)

# Setup checkpointing to save the best model
checkpointer = [keras.callbacks.ModelCheckpoint(filepath='{}/bmodel.h5'.format(SAVE_DIR), monitor='val_acc', mode='max',
                                                save_best_only=True),
                keras.callbacks.EarlyStopping(monitor='val_acc', patience=PATIENCE)]  # Monitor validation accuracy

# Fit the model using the training generator and validate on the validation generator
history = model.fit_generator(G, 7406 // train_batch_size, validation_data=G2,
                              validation_steps=int(924 / val_batch_size), epochs=300)

# Plot training and validation accuracy
plt.figure()
plt.plot(history.history['acc'])  # Training accuracy
plt.plot(history.history['val_acc'])  # Validation accuracy
plt.title('Model accuracy {}'.format(exp_id))
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.grid(True)
plt.savefig('{}/Accuracy.png'.format(SAVE_DIR))  # Save accuracy plot

# Save accuracy results to a text file
print("Starting to save accuracy file")
txt_name2 = "acc_{}.txt".format(exp_id)
this_file = open(SAVE_DIR + "/" + txt_name2, "w")
this_file.write("acc")
this_file.write(str(history.history['acc']))
this_file.write("\n")
this_file.write("val_acc")
this_file.write(str(history.history['val_acc']))
this_file.write("\n")
this_file.close()
print("END END END")

# Plot training and validation loss
plt.figure()
plt.plot(history.history['loss'])  # Training loss
plt.plot(history.history['val_loss'])  # Validation loss
plt.title('Model loss {}'.format(exp_id))
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.grid(True)
plt.savefig('{}/Loss.png'.format(SAVE_DIR))  # Save loss plot

# Save loss results to a text file
print("Starting to save loss file")
txt_name = "loss_{}.txt".format(exp_id)
this_file = open(SAVE_DIR + "/" + txt_name, "w")
this_file.write("loss")
this_file.write(str(history.history['loss']))
this_file.write("\n")
this_file.write("val_loss")
this_file.write(str(history.history['val_loss']))
this_file.write("\n")
this_file.close()

print("Result plotting and saving complete, check the directory {}".format(SAVE_DIR))
print("Model saved in {}".format(SAVE_DIR))

# Calculate elapsed time
elapsed_time = time.time() - start
print(elapsed_time)
print("Total time taken: {} minutes".format(elapsed_time / 60))
