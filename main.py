from EmoSys import demo
from model import train_model, valid_model
import tensorflow.compat.v1 as tf
from absl import flags

tf.compat.v1.disable_v2_behavior()

FLAGS = flags.FLAGS

flags.DEFINE_string('MODE', 'EmoSys', 'Set program to run in different mode: train, valid, EmoSys.')
flags.DEFINE_string('checkpoint_dir', './ckpt', 'Path to model file.')
flags.DEFINE_string('train_data', './data/fer2013/fer2013.csv', 'Path to training data.')
flags.DEFINE_string('valid_data', './valid_sets/', 'Path to validation data.')
flags.DEFINE_boolean('show_box', False, 'If true, the results will show detection box')


def main(argv):
    del argv  # Unused argument
    assert FLAGS.MODE in ('train', 'valid', 'EmoSys')

    if FLAGS.MODE == 'EmoSys':
        demo(FLAGS.checkpoint_dir, FLAGS.show_box)
    elif FLAGS.MODE == 'train':
        train_model(FLAGS.train_data)
    elif FLAGS.MODE == 'valid':
        valid_model(FLAGS.checkpoint_dir, FLAGS.valid_data)


if __name__ == '__main__':
    tf.app.run(main)
