import keras

model = keras.models.load_model("saved_models/1578953293-combined.h5")
disc = keras.models.load_model("saved_models/1578953293-discriminator.h5")
gen = keras.models.load_model("saved_models/1578953293-generator.h5")