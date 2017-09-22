import tensorflow as tf
import json

tf.app.flags.DEFINE_integer('number', 10, 'Integer example')
tf.app.flags.DEFINE_string('name', 'model_name', 'String example')

FLAGS = tf.app.flags.FLAGS

# Call first
num = FLAGS.number
name = FLAGS.name

config = dict(FLAGS.__flags.items())
print (config)

with open('config.json', 'w') as file:
    json.dump(config, file)

with open('config.json') as data_file:    
    saved = json.load(data_file)
print (type(saved))

for k, v in saved.items():
    print (k, v)

print (saved['number'])
print (saved['name'])

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

myconfig = Config(**saved)
print (myconfig.number)
print (myconfig.name)