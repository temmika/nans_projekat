import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from model.cnn_model import build_model

# Parametri
batch_size = 32
img_size = (128, 128)
epochs = 10

train_dir = 'data/data_processing/train'
val_dir = 'data/data_processing/val'

# Augmentacija
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2, horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

val_generator = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

# Model
model = build_model(input_shape=(128, 128, 3), num_classes=5)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Treniranje
model.fit(train_generator, epochs=epochs, validation_data=val_generator)

# Čuvanje modela
model.save('fruit_model.h5')
print("Model sačuvan kao fruit_model.h5")
