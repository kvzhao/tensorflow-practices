import tensorflow as tf
import json

tf.app.flags.DEFINE_integer('number', 10, 'Integer example')
tf.app.flags.DEFINE_string('name', 'model_name', 'String example')

FLAGS = tf.app.flags.FLAGS

config = dict(FLAGS.__flags.items())
print (config)

with open('config.json', 'w') as file:
    json.dump(config, file)

# SHIT, it's empty! why?