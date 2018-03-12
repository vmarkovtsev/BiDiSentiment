import argparse
import csv
import gzip
import io
import logging
import os
import random
import sys
from typing import Tuple, List

import numpy


def setup():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", help="Path to the train dataset in CSV format.")
    parser.add_argument("-l", "--layers", default="512,256",
                        help="Layers configuration: number of neurons on each layer separated by "
                             "comma.")
    parser.add_argument("-m", "--length", type=int, default=180, help="RNN sequence length.")
    parser.add_argument("-b", "--batch-size", type=int, default=128, help="Batch size.")
    parser.add_argument("-e", "--epochs", type=int, default=10, help="Number of epochs.")
    parser.add_argument("-t", "--type", default="LSTM",
                        choices=("GRU", "LSTM", "CuDNNLSTM", "CuDNNGRU"),
                        help="Recurrent layer type to use.")
    parser.add_argument("-v", "--validation", type=float, default=0.2,
                        help="Fraction of the dataset to use for validation.")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to the resulting Tensorflow graph.")
    parser.add_argument("--optimizer", default="Adam", choices=("RMSprop", "Adam"),
                        help="Optimizer to apply.")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout ratio.")
    parser.add_argument("--lr", default=0.001, type=float, help="Learning rate.")
    parser.add_argument("--decay", default=0.00005, type=float, help="Learning rate decay.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--devices", default="0,1", help="Devices to use. Empty means CPU.")
    parser.add_argument("--tensorboard", default="tb_logs",
                        help="TensorBoard output logs directory.")
    parser.add_argument("--snapshot", help="Keras model snapshot to load.")
    logging.basicConfig(level=logging.INFO)
    args = parser.parse_args()
    numpy.random.seed(args.seed)
    random.seed(args.seed)
    return args


def read_dataset(path: str, sequence_length: int, batch_size: int) \
        -> Tuple[List[numpy.ndarray], numpy.ndarray]:
    log = logging.getLogger("reader")
    if path.endswith(".gz"):
        fin = io.TextIOWrapper(gzip.open(path), newline="")
    else:
        fin = open(path, newline="")
    try:
        size = sum(1 for _ in csv.reader(fin))
        rounded_size = size - size % batch_size
        log.info("Size: %d -> %d", size, rounded_size)
        size = rounded_size
        fin.seek(0)
        dataset = [numpy.zeros((size, sequence_length), dtype=numpy.uint8) for _ in range(2)]
        labels = numpy.zeros((size, 2), dtype=numpy.float32)
        for i, row in enumerate(csv.reader(fin)):
            if i % 1000 == 0:
                sys.stderr.write("%d\r" % i)
            if i >= size:
                break
            labels[i][int(row[0])] = 1
            bintext = row[1].strip().encode("utf-8")[-sequence_length:]
            dataset[0][i][-len(bintext):] = list(bintext)
            dataset[1][i][-len(bintext):] = list(reversed(bintext))
    finally:
        sys.stderr.write("\n")
        fin.close()
    return dataset, labels


def config_keras():
    import tensorflow as tf
    from keras import backend
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    backend.tensorflow_backend.set_session(tf.Session(config=config))


def create_char_rnn_model(args: argparse.Namespace):
    # this late import prevents from loading Tensorflow too soon
    import tensorflow as tf
    tf.set_random_seed(args.seed)
    from keras import layers, models, initializers, optimizers, metrics
    log = logging.getLogger("model")
    if args.devices:
        dev1, dev2 = ("/gpu:" + dev for dev in args.devices.split(","))
    else:
        dev1 = dev2 = "/cpu:0"

    def add_rnn(device):
        with tf.device(device):
            input = layers.Input(batch_shape=(args.batch_size, args.length), dtype="uint8")
            log.info("Added %s", input)
            embedding = layers.Embedding(
                256, 256, embeddings_initializer=initializers.Identity(), trainable=False)(input)
            log.info("Added %s", embedding)
        layer = embedding
        layer_sizes = [int(n) for n in args.layers.split(",")]
        for i, nn in enumerate(layer_sizes):
            with tf.device(device):
                layer_type = getattr(layers, args.type)
                ret_seqs = (i < len(layer_sizes) - 1)
                try:
                    layer = layer_type(nn, return_sequences=ret_seqs, implementation=2)(layer)
                except TypeError:
                    # implementation kwarg is not present in CuDNN layers
                    layer = layer_type(nn, return_sequences=ret_seqs)(layer)
                log.info("Added %s", layer)
            if args.dropout > 0:
                layer = layers.Dropout(args.dropout)(layer)
                log.info("Added %s", layer)
        return input, layer

    forward_input, forward_output = add_rnn(dev1)
    reverse_input, reverse_output = add_rnn(dev2)
    with tf.device(dev1):
        merged = layers.Concatenate()([forward_output, reverse_output])
        log.info("Added %s", merged)
        dense = layers.Dense(2, activation="softmax")
        decision = dense(merged)
        log.info("Added %s", decision)
    optimizer = getattr(optimizers, args.optimizer)(lr=args.lr, decay=args.decay)
    log.info("Added %s", optimizer)
    model = models.Model(inputs=[forward_input, reverse_input], outputs=[decision])
    log.info("Compiling...")
    model.compile(optimizer=optimizer, loss="binary_crossentropy",
                  metrics=[metrics.binary_accuracy])
    log.info("Done")
    return model


def train_char_rnn_model(model, dataset: Tuple[numpy.ndarray, numpy.ndarray],
                         args: argparse.Namespace):
    from keras import callbacks

    if args.length % 2 != 0:
        raise ValueError("--length must be even")
    log = logging.getLogger("train")
    log.info("model.fit")
    tensorboard = callbacks.TensorBoard(log_dir=args.tensorboard)
    checkpoint = callbacks.ModelCheckpoint(
        os.path.join(args.tensorboard, "checkpoint_{epoch:02d}_{val_loss:.3f}.hdf5"),
        save_best_only=True)

    class LRPrinter(callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            from keras import backend
            lr = self.model.optimizer.lr
            decay = self.model.optimizer.decay
            iterations = self.model.optimizer.iterations
            lr_with_decay = lr / (1. + decay * backend.cast(iterations, backend.dtype(decay)))
            print("Learning rate:", backend.eval(lr_with_decay))

    model.fit(dataset[0], dataset[1],
              batch_size=args.batch_size, validation_split=args.validation,
              epochs=args.epochs, callbacks=[tensorboard, checkpoint, LRPrinter()])


def export_model(model, path: str):
    from keras import backend
    import tensorflow as tf
    from tensorflow.python.framework import graph_util, graph_io

    log = logging.getLogger("export")
    log.info("Exporting %s to %s", model, path)
    session = backend.get_session()
    tf.identity(model.outputs[0], name="output")
    graph_def = session.graph.as_graph_def()
    # reset the devices
    for node in graph_def.node:
        node.device = ""
    constant_graph = graph_util.convert_variables_to_constants(session, graph_def, ["output"])
    graph_io.write_graph(constant_graph, *os.path.split(path), as_text=False)


def main():
    args = setup()
    try:
        if not args.snapshot:
            if args.validation == 0:
                round_size = args.batch_size
            else:
                round_size = int(args.batch_size / args.validation)
            dataset = read_dataset(args.input, args.length, round_size)
            config_keras()
            model_char = create_char_rnn_model(args)
            train_char_rnn_model(model_char, dataset, args)
            del dataset
        else:
            from keras.models import load_model
            model_char = load_model(args.snapshot)
        export_model(model_char, args.output)
        del model_char
    finally:
        from keras import backend
        backend.clear_session()

if __name__ == "__main__":
    sys.exit(main())
