import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

# Execute this to avoid internal tf error
physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

###

from training_images.training_data_provider import TrainingDataProvider
provider = TrainingDataProvider()

facial_image_generator = provider.get_all_training_images_in_batches_from_disk

###

from gan import HR_DCGAN_Faces, DCGAN_Faces

gan = DCGAN_Faces(show_training_results=False)

gan.set_training_data(facial_image_generator)

###

gan.train(epochs=50)

### 

gan.export()
gan.generate_gif()