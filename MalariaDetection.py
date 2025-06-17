# ===============================
# Data Preparation
# ===============================

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np

def format_image(image, label):
    image = tf.image.resize(image, (224, 224)) / 255.0
    return image, label

(raw_train, raw_val, raw_test), metadata = tfds.load(
    'malaria',
    split=["train[:80%]", "train[80%:90%]", "train[90%:]"],
    with_info=True,
    as_supervised=True
)

NO_OF_EXAMPLES = metadata.splits['train'].num_examples
NO_OF_CLASSES = metadata.features['label'].num_classes
NO_OF_BATCHES = 64
NO_OF_SHUFFLES = NO_OF_EXAMPLES // 4
AUTOTUNE = tf.data.AUTOTUNE

train_batches = raw_train.shuffle(NO_OF_SHUFFLES).map(format_image).batch(NO_OF_BATCHES).prefetch(AUTOTUNE)
val_batches = raw_val.map(format_image).batch(NO_OF_BATCHES).prefetch(AUTOTUNE)
test_batches = raw_test.map(format_image).batch(1)

# Preview one batch shape
for image_batch, label_batch in train_batches.take(1):
    pass
print("Image batch shape:", image_batch.shape)

# ===============================
# Model Definition and Training
# ===============================

model_selection = ("mobilenet_v2", 224, 1280)
handle_base, pixels, fvsize = model_selection
model_handle = f"https://tfhub.dev/google/tf2-preview/{handle_base}/feature_vector/4"

model = tf.keras.Sequential([
    hub.KerasLayer(model_handle, input_shape=(pixels, pixels, 3), output_shape=[fvsize], trainable=False),
    tf.keras.layers.Dense(NO_OF_CLASSES, activation='softmax')
])

model.summary()

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=['accuracy']
)

model.fit(
    train_batches,
    epochs=3,
    validation_data=val_batches
)

# ===============================
# Saving and Converting Model
# ===============================

export_dir = "SavedModels/Model1"
tf.saved_model.save(model, export_dir)

import pathlib

converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
tflite_model = converter.convert()

tflite_model_dir = pathlib.Path("/tmp")
tflite_model_file = tflite_model_dir / "model1.tflite"
tflite_model_file.write_bytes(tflite_model)

# ===============================
# TFLite Model Evaluation
# ===============================

from tqdm import tqdm

interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']

correct = 0
total = 0

for image, label in tqdm(test_batches):
    interpreter.set_tensor(input_index, image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_index)
    prediction = np.argmax(output)
    actual = label.numpy()[0]
    if prediction == actual:
        correct += 1
    total += 1

accuracy = correct / total
print("Accuracy: {:.2f}%".format(accuracy * 100))

# ===============================
# Quantization and Evaluation
# ===============================

def get_tflite_model(optimization):
    converter = tf.lite.TFLiteConverter.from_saved_model(export_dir)
    converter.optimizations = [optimization]

    def representative_data_gen():
        for input_value, _ in test_batches.take(100):
            yield [input_value]

    converter.representative_dataset = representative_data_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    return converter.convert()

def save_tflite_model(myModel, name):
    tflite_model_file = pathlib.Path("/tmp") / f"{name}.tflite"
    tflite_model_file.write_bytes(myModel)

def evaluate_model(myModel):
    interpreter = tf.lite.Interpreter(model_content=myModel)
    interpreter.allocate_tensors()
    input_index = interpreter.get_input_details()[0]['index']
    output_index = interpreter.get_output_details()[0]['index']
    correct = 0
    total = 0

    for image, label in tqdm(test_batches):
        interpreter.set_tensor(input_index, image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)
        prediction = np.argmax(output)
        actual = label.numpy()[0]
        if prediction == actual:
            correct += 1
        total += 1

    return correct / total

# ===============================
# Quantization Variants
# ===============================

tflite_model1 = get_tflite_model(tf.lite.Optimize.DEFAULT)
tflite_model2 = get_tflite_model(tf.lite.Optimize.OPTIMIZE_FOR_LATENCY)
tflite_model3 = get_tflite_model(tf.lite.Optimize.OPTIMIZE_FOR_SIZE)
tflite_model4 = get_tflite_model(tf.lite.Optimize.EXPERIMENTAL_SPARSITY)

save_tflite_model(tflite_model1, "tfmodel1")
save_tflite_model(tflite_model2, "tfmodel2")
save_tflite_model(tflite_model3, "tfmodel3")
save_tflite_model(tflite_model4, "tfmodel4")

# ===============================
# Quantized Model Evaluation Results
# ===============================

print("Accuracy1 : {:.2f}%".format(evaluate_model(tflite_model1) * 100))
print("Accuracy2 : {:.2f}%".format(evaluate_model(tflite_model2) * 100))
print("Accuracy3 : {:.2f}%".format(evaluate_model(tflite_model3) * 100))
print("Accuracy4 : {:.2f}%".format(evaluate_model(tflite_model4) * 100))
