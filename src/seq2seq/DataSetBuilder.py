import random
from multiprocessing import Pool

from seq2seq.Evaluator import Evaluator
from utils import batcher, dumper
from utils.wrapper import *
from variables.embeddings import *
from variables.paths import *
from variables.syntax import *
from variables.tags import *
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
    def build_samples(
            doc, evaluate, filtrator, num_samples: int, num_functions: int, num_genetic_cycles: int,
            noise_depth: int
    ):
        indexes = {}
        inputs_sizes = {}
        for label, embeddings in doc:
            inputs_sizes[label] = []
            indexes[label] = embeddings + [Embeddings.get_index(PAD) for _ in range(INPUT_SIZE - len(embeddings))]
            inputs_sizes[label].append(len(embeddings))
        samples = [[] for _ in range(num_functions)]
        for _ in range(num_samples // num_functions):
            for functions in range(num_functions):
                sample = []
                expected_functions = functions + 1
                expected = OUTPUT_SIZE
                while True:
                    func = Functions[random.randrange(len(Functions))]
                    expected -= func.arguments + 1
                    if expected <= 0 or expected_functions == 0:
                        sample.extend([Tokens.END.embedding] * (func.arguments + 1 + expected))
                        break
                    else:
                        expected_functions -= 1
                        sample.append(func.embedding)
                        for _ in range(func.arguments):
                            constant = Constants[random.randrange(len(Constants))]
                            sample.append(constant.embedding)
                sample_pair = (sample, evaluate(indexes, sample))
                samples[functions].append(sample_pair)
        for expected_functions in range(num_functions):
            current_samples = samples[expected_functions]
            random.shuffle(current_samples)
            args = (indexes, current_samples, evaluate, filtrator, num_genetic_cycles, noise_depth)
            samples[expected_functions] = DataSetBuilder.noise_samples(*args)
        result = set()
        for _samples in samples:
            result.update(((tuple(tuple(one_hot) for one_hot in sample), evaluate) for sample, evaluate in _samples))
        return (indexes, inputs_sizes), list(result)

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
                noised_samples.append((sample, evaluate(inputs, sample)))
            noised_samples = filtrator(noised_samples, num_samples, lambda x: x[1])
        return noised_samples

    @staticmethod
    def build_most_different_samples(doc, evaluate, num_samples: int, num_functions: int, genetic_cycles: int,
                                     noise_depth: int):
        inputs, samples = DataSetBuilder.build_samples(doc, evaluate, DataSetBuilder.most_different,
                                                       num_samples, num_functions, genetic_cycles, noise_depth)
        return [(inputs, sample) for sample in samples]

    @staticmethod
    def build_best_samples(doc, evaluate, num_samples: int, num_functions: int, genetic_cycles: int, noise_depth: int):
        inputs, samples = DataSetBuilder.build_samples(doc, evaluate, DataSetBuilder.best,
                                                       num_samples, num_functions, genetic_cycles, noise_depth)
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
                inputs_sizes[label].extend(inp_sizes[label])
            samples.append(sample)
            evaluations.append(evaluation)
        for label in PARTS:
            indexes[label] = np.transpose(indexes[label], axes=(1, 0))
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
    def build_samples_data_set(sample_builder, save_path: str, num_samples: int, num_functions: int,
                               genetic_cycles: int, noise_depth: int):
        methods = dumper.load(VEC_METHODS)
        with Pool() as pool:
            docs = pool.map(DataSetBuilder.indexes, methods)
            docs_baskets = batcher.throwing(docs, [INPUT_SIZE])
            docs = docs_baskets[INPUT_SIZE]
            data = ((doc, Evaluator.evaluate, num_samples, num_functions, genetic_cycles, noise_depth) for doc in docs)
            raw_samples = pool.starmap(sample_builder, data)
            samples = [sample for samples in raw_samples for sample in samples]
            random.shuffle(samples)
            batches = batcher.chunks(samples, BATCH_SIZE)
            batches = pool.map(DataSetBuilder.refactor_batch, batches)
        dumper.dump(batches, save_path)

    @staticmethod
    @trace
    def build_most_different_data_set(save_path: str, num_samples: int, num_functions: int, genetic_cycles: int,
                                      noise_depth: int):
        DataSetBuilder.build_samples_data_set(DataSetBuilder.build_most_different_samples, save_path,
                                              num_samples, num_functions, genetic_cycles, noise_depth)

    @staticmethod
    @trace
    def build_best_data_set(save_path: str, num_samples: int, num_functions: int, genetic_cycles: int,
                            noise_depth: int):
        DataSetBuilder.build_samples_data_set(DataSetBuilder.build_best_samples, save_path, num_samples, num_functions,
                                              genetic_cycles, noise_depth)

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
