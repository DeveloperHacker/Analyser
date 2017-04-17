import random
from multiprocessing import Pool

from seq2seq.Evaluator import Evaluator
from utils import batcher, dumper
from utils.wrapper import *
from variables.embeddings import *
from variables.tags import *
from variables.paths import *
from variables.syntax import *
from variables.train import *


class DataSetBuilder:
    @staticmethod
    def most_different(samples: list, quantity: int, key: callable = None):
        NUM_COLUMNS = 4

        minimum = key(min(samples, key=key))
        maximum = key(max(samples, key=key))
        minimum -= minimum * 1e-4
        maximum += maximum * 1e-4
        delta = (maximum - minimum) / NUM_COLUMNS
        baskets = batcher.hist(samples, list(np.arange(minimum, maximum, delta))[1:], key=key)
        maximum = len(key(max(baskets.items(), key=lambda x: len(key(x)))))
        line = None
        for i in range(maximum):
            num_samples = sum([min(len(samples), i) for _, samples in baskets.items()])
            if num_samples >= quantity:
                line = i
                break
        assert line is not None, "NS: {} < M: {}".format(len(samples), quantity)
        samples = []
        for _, basket in baskets.items():
            random.shuffle(basket)
            samples.extend(basket[:min(len(basket), line)])
        return samples[:quantity]

    @staticmethod
    def best(samples: list, quantity: int, key: callable = None):
        samples.sort(key=key)
        return samples[:quantity]

    @staticmethod
    def build_samples(doc, evaluate, filtrator, num_samples: int, num_genetic_cycles: int, noise_depth: int):
        ARGUMENT = 1
        FUNCTION = 2

        indexes = {}
        inputs_sizes = {}
        for label, embeddings in doc:
            indexes[label] = []
            inputs_sizes[label] = []
            for _ in range(INPUT_SIZE):
                indexes[label].append([])
            line = embeddings + [Embeddings.get_index(PAD) for _ in range(INPUT_SIZE - len(embeddings))]
            inputs_sizes[label].append(len(embeddings))
            for i, embedding in enumerate(line):
                indexes[label][i].append(embedding)
        samples = []
        for _ in range(num_samples):
            sample = []
            state = FUNCTION
            expected = OUTPUT_SIZE
            arguments = None
            num_functions = random.randrange(expected // 2)
            while True:
                if state == FUNCTION:
                    i = random.randrange(len(Functions))
                    arguments = Functions[i].arguments
                    expected -= arguments + 1
                    if expected <= 0 or num_functions == 0:
                        sample.extend([Tokens.END.embedding] * (arguments + expected + 1))
                        break
                    sample.append(Functions[i].embedding)
                    num_functions -= 1
                    state = ARGUMENT
                elif state == ARGUMENT:
                    i = random.randrange(len(Constants))
                    constant = Constants[i]
                    sample.append(constant.embedding)
                    arguments -= 1
                    if arguments == 0:
                        state = FUNCTION
            samples.append((sample, evaluate(indexes, np.expand_dims(sample, axis=1))[0]))
        random.shuffle(samples)
        samples = DataSetBuilder.noise_samples(indexes, samples, evaluate, filtrator, num_genetic_cycles, noise_depth)
        return (indexes, inputs_sizes), samples

    @staticmethod
    def noise_samples(inputs, samples, evaluate, filtrator, num_genetic_cycles: int, noise_depth: int):
        noised_samples = list(samples)
        num_samples = len(noised_samples)
        for i in range(num_genetic_cycles):
            for j in range(num_samples):
                sample = noised_samples[j][0][::]
                indexes = random.sample(list(np.arange(len(sample))), noise_depth)
                for index in indexes:
                    n = random.randrange(NUM_TOKENS)
                    sample[index] = Tokens.get(n).embedding
                noised_samples.append((sample, evaluate(inputs, np.expand_dims(sample, axis=1))[0]))
            noised_samples = filtrator(noised_samples, num_samples, lambda x: x[1])
        return noised_samples

    @staticmethod
    def build_most_different_samples(doc, evaluate, num_samples: int, genetic_cycles: int, noise_depth: int):
        inputs, samples = DataSetBuilder.build_samples(doc, evaluate, DataSetBuilder.most_different,
                                                       num_samples, genetic_cycles, noise_depth)
        return [(inputs, sample) for sample in samples]

    @staticmethod
    def build_best_samples(doc, evaluate, num_samples: int, genetic_cycles: int, noise_depth: int):
        inputs, samples = DataSetBuilder.build_samples(doc, evaluate, DataSetBuilder.most_different,
                                                       num_samples, genetic_cycles, noise_depth)
        return [(inputs, sample) for sample in samples]

    @staticmethod
    def refactor_batch(batch: list):
        indexes = {label: [] for label in PARTS}
        inputs_sizes = {label: [] for label in PARTS}
        samples = []
        evaluations = []
        for (inp, inp_sizes), (sample, evaluation) in batch:
            for label in PARTS:
                indexes[label].append(inp[label])
                inputs_sizes[label].append(inp_sizes[label])
            samples.append(sample)
            evaluations.append(evaluation)
        for label in PARTS:
            indexes[label] = np.transpose(np.asarray(indexes[label]), axes=(1, 0, 2))
            indexes[label] = np.squeeze(indexes[label], axis=(2,))
            inputs_sizes[label] = np.squeeze(np.asarray(inputs_sizes[label]), axis=(1,))
        samples = np.transpose(np.asarray(samples), axes=(1, 0, 2))
        return indexes, inputs_sizes, samples, evaluations

    @staticmethod
    def indexes(method):
        doc = []
        for label, (embeddings, text) in method:
            embeddings = [Embeddings.get_index(word) for embedding, word in zip(embeddings, text)]
            doc.append((label, embeddings))
        return doc

    @staticmethod
    def build_samples_data_set(sample_builder, save_path: str, num_samples: int, genetic_cycles: int, noise_depth: int):
        methods = dumper.load(VEC_METHODS)
        with Pool() as pool:
            docs = pool.map(DataSetBuilder.indexes, methods)
            docs_baskets = batcher.throwing(docs, [INPUT_SIZE])
            docs = docs_baskets[INPUT_SIZE]
            data = ((doc, Evaluator.evaluate, num_samples, genetic_cycles, noise_depth) for doc in docs)
            raw_samples = pool.starmap(sample_builder, data)
            samples = [sample for samples in raw_samples for sample in samples]
            random.shuffle(samples)
            batches = batcher.chunks(samples, BATCH_SIZE)
            batches = pool.map(DataSetBuilder.refactor_batch, batches)
        dumper.dump(batches, save_path)

    @staticmethod
    @trace
    def build_most_different_data_set(save_path: str, num_samples: int, genetic_cycles: int, noise_depth: int):
        DataSetBuilder.build_samples_data_set(DataSetBuilder.build_most_different_samples, save_path,
                                              num_samples, genetic_cycles, noise_depth)

    @staticmethod
    @trace
    def build_best_data_set(save_path: str, num_samples: int, genetic_cycles: int, noise_depth: int):
        DataSetBuilder.build_samples_data_set(DataSetBuilder.build_best_samples, save_path,
                                              num_samples, genetic_cycles, noise_depth)

    @staticmethod
    def indexes(method):
        doc = []
        for label, (embeddings, text) in method:
            indexes = [Embeddings.get_index(word) for embedding, word in zip(embeddings, text)]
            doc.append((label, indexes))
        return doc

    @staticmethod
    def build_batch(batch: list):
        indexes = {label: [] for label in PARTS}
        inputs_sizes = {label: [] for label in PARTS}
        for datum in batch:
            for label, _indexes in datum:
                _inputs_sizes = len(_indexes)
                _indexes += [Embeddings.get_index(PAD) for _ in range(INPUT_SIZE - len(_indexes))]
                indexes[label].append(_indexes)
                inputs_sizes[label].append(_inputs_sizes)
        indexes = {label: np.transpose(np.asarray(indexes[label]), axes=(1, 0)) for label in PARTS}
        return indexes, inputs_sizes

    @staticmethod
    @trace
    def build_vectorized_methods_data_set(save_path: str):
        methods = dumper.load(VEC_METHODS)
        with Pool() as pool:
            docs = pool.map(DataSetBuilder.indexes, methods)
            docs_baskets = batcher.throwing(docs, [INPUT_SIZE])
            docs = docs_baskets[INPUT_SIZE]
            random.shuffle(docs)
            batches = batcher.chunks(docs, BATCH_SIZE)
            batches = pool.map(DataSetBuilder.build_batch, batches)
        batches = [batch for batch in batches if len(list(batch[1].values())[0]) == BATCH_SIZE]
        random.shuffle(batches)
        dumper.dump(batches, save_path)
