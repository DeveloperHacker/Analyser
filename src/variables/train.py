WORD2VEC_EPOCHS = 500
CONTRACT_EPOCHS = 15
ANALYSER_EPOCHS = 10
Q_FUNCTION_EPOCHS = 10
MÜNCHHAUSEN_PRETRAIN_EPOCHS = 30
MÜNCHHAUSEN_TRAIN_EPOCHS = 50
MÜNCHHAUSEN_RUNS = 2000

LPR = {5: 70, 10: 20, 20: 5}
MGS = 1_000_000

TRAIN_SET = 0.9
VALIDATION_SET = 0.1

EMBEDDING_SIZE = 100
WINDOW = 7
INPUT_STATE_SIZE = 150
OUTPUT_STATE_SIZE = 150
BATCH_SIZE = 20
INPUT_SIZE = 30
OUTPUT_SIZE = 10

CONTAINS_WEIGHT = 1
VARIANCE_WEIGHT = 1
Q_WEIGHT = 1
DIFF_WEIGHT = 1
L2_WEIGHT = 0
OH_WEIGHT = 1
Q_INERTNESS = 0
DIFF_INERTNESS = 0

BLOCK_SIZE = 1000

REGULARIZATION_VARIABLES = (
    "analyser/encoder/head/encoder/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "analyser/encoder/head/encoder/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "analyser/encoder/param/encoder/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "analyser/encoder/param/encoder/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "analyser/encoder/variable/encoder/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "analyser/encoder/variable/encoder/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "analyser/encoder/return/encoder/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "analyser/encoder/return/encoder/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "analyser/decoder/decoder/weights:0",
    "analyser/decoder/decoder/attention_decoder/weights:0",
    "analyser/decoder/decoder/attention_decoder/gru_cell/candidate/weights:0",
    "analyser/decoder/decoder/attention_decoder/AttnOutputProjection/weights:0",
    "analyser/linear/weights:0",
    "q-function/encoder/encoder/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "q-function/encoder/encoder/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "q-function/decoder/decoder/weights:0",
    "q-function/decoder/decoder/attention_decoder/weights:0",
    "q-function/decoder/decoder/attention_decoder/gru_cell/candidate/weights:0",
    "q-function/decoder/decoder/attention_decoder/AttnOutputProjection/weights:0",
    "q-function/sigmoid/weights:0",
)
