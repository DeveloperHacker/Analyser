import itertools
import os
import random
import time

import numpy as np
import tensorflow as tf
from live_plotter.proxy.ProxyFigure import ProxyFigure
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.rnn_cell_impl import GRUCell

from constants.analyser import *
from constants.embeddings import WordEmbeddings, TokenEmbeddings, NUM_TOKENS, PAD
from constants.paths import RESOURCES, ANALYSER, ANALYSER_METHODS
from constants.tags import PARTS, NEXT
from seq2seq.Net import Net
from seq2seq.analyser_rnn import sequence_input, input_projection, analysing_loss, sequence_output, tree_output
from utils import Dumper
from utils.Formatter import Formatter
from utils.wrapper import trace


class AnalyserNet(Net):
    @trace
    def __init__(self):
        super().__init__("analyser", ANALYSER)
        with vs.variable_scope(self.name):
            self.inputs = []
            self.embeddings = []
            self.inputs_sizes = []
            embeddings = tf.constant(np.asarray(WordEmbeddings.idx2emb()))
            for label in PARTS:
                with vs.variable_scope(label):
                    indexes = tf.placeholder(tf.int32, [BATCH_SIZE, None], "indexes")
                    self.embeddings.append(tf.gather(embeddings, indexes))
                    self.inputs.append(indexes)
                    self.inputs_sizes.append(tf.placeholder(tf.int32, [BATCH_SIZE], "input_sizes"))
            self.root_time_steps = tf.placeholder(tf.int32, [], "root_time_steps")
            self.output_time_steps = tf.placeholder(tf.int32, [], "output_time_steps")
            self.depth = tf.placeholder(tf.int32, [], "depth")
            cells_fw = [GRUCell(ENCODER_STATE_SIZE) for _ in range(NUM_ENCODERS)]
            cells_bw = [GRUCell(ENCODER_STATE_SIZE) for _ in range(NUM_ENCODERS)]
            root_cells = [GRUCell(ROOT_STATE_SIZE) for _ in range(NUM_DECODERS)]
            sequence_cells = [GRUCell(SEQUENCE_STATE_SIZE) for _ in range(NUM_DECODERS)]
            projection = input_projection(self.embeddings, INPUT_SIZE, tf.float32)
            attention_states = sequence_input(cells_bw, cells_fw, projection, self.inputs_sizes, tf.float32)
            output_type = os.environ['OUTPUT_TYPE']
            if output_type == "tree":
                self.outputs_logits, self.outputs = tree_output(
                    attention_states,
                    root_cells,
                    self.root_time_steps,
                    ROOT_NUM_HEADS,
                    NUM_TOKENS,
                    self.depth,
                    tf.float32)
            elif output_type in ("bfs_sequence", "dfs_sequence"):
                self.outputs_logits, self.outputs = sequence_output(
                    attention_states,
                    root_cells,
                    self.root_time_steps,
                    ROOT_NUM_HEADS,
                    sequence_cells,
                    self.output_time_steps,
                    SEQUENCE_NUM_HEADS,
                    NUM_TOKENS,
                    tf.float32)
            self.top_outputs = tf.nn.top_k(self.outputs, TOP)
            self.outputs_targets = tf.placeholder(tf.int32, [BATCH_SIZE, None, None], "outputs_targets")
            self.scope = vs.get_variable_scope().name
            self.losses = analysing_loss(((self.outputs_targets, self.outputs_logits),), self.get_variables())
            self.loss = tf.reduce_mean(self.losses)
        self.optimizer = tf.train.AdamOptimizer().minimize(self.loss)
        self.data_set = Dumper.pkl_load(ANALYSER_METHODS)

    def get_data_set(self) -> (list, list, list):
        data_set = list(self.data_set)
        data_set_length = len(self.data_set)
        not_allocated = data_set_length
        test_set_length = min(not_allocated, int(data_set_length * TEST_SET))
        not_allocated -= test_set_length
        train_set_length = min(not_allocated, int(data_set_length * TRAIN_SET))
        not_allocated -= train_set_length
        validation_set_length = min(not_allocated, int(data_set_length * VALIDATION_SET))
        not_allocated -= validation_set_length
        if test_set_length < MINIMUM_DATA_SET_LENGTH:
            args = (test_set_length, MINIMUM_DATA_SET_LENGTH)
            raise ValueError("Length of the test set is very small, length = %d < %d" % args)
        if train_set_length < MINIMUM_DATA_SET_LENGTH:
            args = (train_set_length, MINIMUM_DATA_SET_LENGTH)
            raise ValueError("Length of the train set is very small, length = %d < %d" % args)
        if validation_set_length < MINIMUM_DATA_SET_LENGTH:
            args = (validation_set_length, MINIMUM_DATA_SET_LENGTH)
            raise ValueError("Length of the validation set is very small, length = %d < %d" % args)
        test_set = data_set[-test_set_length:]
        data_set = data_set[:-test_set_length]
        random.shuffle(data_set)
        train_set = data_set[-train_set_length:]
        data_set = data_set[:-train_set_length]
        validation_set = data_set[-validation_set_length:]
        return train_set, validation_set, test_set

    def build_feed_dict(self, batch) -> dict:
        feed_dict = {}
        inputs, inputs_sizes, outputs, root_time_steps, output_time_steps, depth = batch
        for i, label in enumerate(PARTS):
            feed_dict[self.inputs[i]] = np.asarray(inputs[label]).T
            feed_dict[self.inputs_sizes[i]] = inputs_sizes[label]
        feed_dict[self.root_time_steps] = root_time_steps
        feed_dict[self.output_time_steps] = output_time_steps
        feed_dict[self.depth] = depth
        feed_dict[self.outputs_targets] = outputs
        return feed_dict

    def correct_target(self, feed_dict, outputs) -> dict:
        # return self.nearest_correct(feed_dict, outputs, 0.5)
        return self.greedy_correct(feed_dict, outputs)

    def nearest_correct(self, feed_dict, outputs, fine_weigh):
        outputs_targets = feed_dict[self.outputs_targets]
        embeddings = np.asarray(TokenEmbeddings.idx2emb())
        outputs_targets_values = embeddings[outputs_targets]
        result = []
        for i in range(len(outputs)):
            output = outputs[i]
            output_target = outputs_targets[i]
            output_target_values = outputs_targets_values[i]
            best = None
            best_distance = None
            default_indexes = np.asarray(list(range(len(output_target))))
            for indexes in itertools.permutations(default_indexes):
                perm = output_target_values[list(indexes)]
                distance = np.linalg.norm(perm[1:] - output[1:])
                fine = fine_weigh * np.linalg.norm(default_indexes - indexes)
                distance = distance + fine
                if best_distance is None or distance < best_distance:
                    best = output_target[list(indexes)]
                    best_distance = distance
            result.append(best)
        feed_dict[self.outputs_targets] = np.asarray(result)
        return feed_dict

    def greedy_correct(self, feed_dict, outputs):
        outputs_targets = feed_dict[self.outputs_targets]
        embeddings = np.asarray(TokenEmbeddings.idx2emb())
        outputs_targets_values = embeddings[outputs_targets]
        result = []
        for i in range(len(outputs)):
            output = outputs[i]
            output_target = outputs_targets[i]
            output_target_values = outputs_targets_values[i]
            from_indexes = list(range(len(output_target)))
            to_indexes = list(range(len(output)))
            _result = [None] * len(output)
            while len(from_indexes) > 0:
                index_best_from_index = None
                index_best_to_index = None
                best_distance = None
                for i, from_index in enumerate(from_indexes):
                    for j, to_index in enumerate(to_indexes):
                        from_output = output_target_values[from_index]
                        to_output = output[to_index]
                        distance = np.linalg.norm(from_output[1:] - to_output[1:])
                        if best_distance is None or distance < best_distance:
                            index_best_from_index = i
                            index_best_to_index = j
                            best_distance = distance
                best_from_index = from_indexes[index_best_from_index]
                best_to_index = to_indexes[index_best_to_index]
                del from_indexes[index_best_from_index]
                del to_indexes[index_best_to_index]
                _result[best_to_index] = output_target[best_from_index]
            result.append(_result)
        feed_dict[self.outputs_targets] = np.asarray(result)
        return feed_dict

    @trace
    def pretrain(self):
        pass

    @trace
    def train(self):
        del tf.get_collection_ref('LAYER_NAME_UIDS')[0]  # suppress dummy warning hack

        formatter = Formatter(
            heads=("epoch", "time", "train", "validation"),
            formats=("d", ".4f", ".4f", ".4f"),
            sizes=(10, 20, 20, 20),
            rows=(0, 1, 2, 3),
            height=10)
        figure = ProxyFigure("train")
        train_loss_graph = figure.fill_graph(1, 1, 1, mode="-ob", color="blue", alpha=0.3)
        validation_loss_graph = figure.fill_graph(1, 1, 1, mode="or", color="red", alpha=0.3)
        figure.set_x_label(1, 1, 1, "epoch")
        figure.set_y_label(1, 1, 1, "loss")
        with tf.Session() as session, tf.device('/cpu:0'):
            writer = tf.summary.FileWriter(RESOURCES + "/analyser/summary", session.graph)
            self.reset(session)
            for epoch in range(TRAIN_EPOCHS):
                start = time.time()
                train_set, validation_set, test_set = self.get_data_set()
                for batch in train_set:
                    feed_dict = self.build_feed_dict(batch)
                    outputs = session.run(self.outputs, feed_dict)
                    feed_dict = self.correct_target(feed_dict, outputs)
                    session.run(self.optimizer, feed_dict)
                train_losses = []
                for batch in train_set:
                    feed_dict = self.build_feed_dict(batch)
                    outputs = session.run(self.outputs, feed_dict)
                    feed_dict = self.correct_target(feed_dict, outputs)
                    train_losses.append(session.run(self.loss, feed_dict))
                validation_losses = []
                for batch in validation_set:
                    feed_dict = self.build_feed_dict(batch)
                    outputs = session.run(self.outputs, feed_dict)
                    feed_dict = self.correct_target(feed_dict, outputs)
                    validation_losses.append(session.run(self.loss, feed_dict))
                stop = time.time()
                delay = stop - start
                train_loss = np.mean(train_losses)
                deviation_train_loss = np.sqrt(np.var(train_losses))
                validation_loss = np.mean(validation_losses)
                deviation_validation_loss = np.sqrt(np.var(validation_losses))
                formatter.print(epoch, delay, train_loss, validation_loss)
                train_loss_graph.append(epoch, train_loss, deviation_train_loss)
                validation_loss_graph.append(epoch, validation_loss, deviation_validation_loss)
                figure.draw()
                figure.save(self.get_model_path() + "/train.png")
                writer.flush()
                if np.isnan(train_loss) or np.isnan(validation_loss):
                    raise Net.NaNException()
                self.save(session)
        writer.close()

    @trace
    def test(self):
        formatter = Formatter(
            heads=("loss", "target", *(["output", "prob"] * TOP)),
            formats=(".4f", "s", *(["s", ".4f"] * TOP)),
            sizes=(12, 20, *([20, 12] * TOP)),
            rows=range(2 + 2 * TOP))
        with tf.Session() as session, tf.device('/cpu:0'):
            self.reset(session)
            self.restore(session)
            train_set, validation_set, test_set = self.get_data_set()
            for batch in test_set:
                feed_dict = self.build_feed_dict(batch)
                outputs = session.run(self.outputs, feed_dict)
                feed_dict = self.correct_target(feed_dict, outputs)
                fetches = (self.losses, self.outputs_targets, self.top_outputs, self.inputs)
                ls, outputs_targets, top_outputs, inputs = session.run(fetches, feed_dict)
                top_outputs_indexes = top_outputs.indices
                top_outputs_probs = top_outputs.values
                for l, ot, toi, top, *inps in zip(ls, outputs_targets, top_outputs_indexes, top_outputs_probs, *inputs):
                    formatter.print_head()
                    for ot_i, toi_i, top_i in zip(ot, toi, top):
                        for ot_ij, toi_ij, top_ij in zip(ot_i, toi_i, top_i):
                            ot_ij = TokenEmbeddings.get_token(int(ot_ij))
                            toi_ij = (TokenEmbeddings.get_token(int(i)) for i in toi_ij)
                            formatter.print(l, ot_ij, *(elem for pair in zip(toi_ij, top_ij) for elem in pair))
                        formatter.print_delimiter()
                    for inp, label in zip(inps, PARTS):
                        line = " ".join(WordEmbeddings.get_word(int(i)) for i in inp)
                        lines = (line.strip() for line in line.replace(PAD, " ").split(NEXT))
                        text = "\n".join(line for line in lines if len(line) > 0)
                        formatter.print_appendix(text, label)
                    formatter.print_lower_delimiter()
