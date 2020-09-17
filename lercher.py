import tensorflow as tf

###

from training_images.training_data_provider import TrainingDataProvider
provider = TrainingDataProvider()

facial_image_generator = provider.get_all_training_images_in_batches_from_disk

###

from gan import HR_DCGAN_Faces

gan = HR_DCGAN_Faces(show_training_results=False)

gan.set_training_data(facial_image_generator)

###

gan.train(epochs=50)

### 

gan.generate_gif()
gan.export()