WORD2VEC_EPOCHS = 500
CONTRACT_EPOCHS = 15
ANALYSER_EPOCHS = 10
Q_FUNCTION_EPOCHS = 10
MÜNCHHAUSEN_EPOCHS = 20
MÜNCHHAUSEN_RUNS = 20
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
L2_WEIGHT = 0.0001
OH_WEIGHT = 1

BLOCK_SIZE = 100

REGULARIZATION_VARIABLES = (
    "javadoc-encoder/head/encoder/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "javadoc-encoder/head/encoder/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "javadoc-encoder/param/encoder/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "javadoc-encoder/param/encoder/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "javadoc-encoder/variable/encoder/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "javadoc-encoder/variable/encoder/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "javadoc-encoder/return/encoder/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "javadoc-encoder/return/encoder/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "contract-decoder/decoder/weights:0",
    "contract-decoder/decoder/attention_decoder/weights:0",
    "contract-decoder/decoder/attention_decoder/gru_cell/candidate/weights:0",
    "contract-decoder/decoder/attention_decoder/AttnOutputProjection/weights:0",
    "contract-linear/weights:0",
    "evaluation-encoder/encoder/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "evaluation-encoder/encoder/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "evaluation-decoder/decoder/weights:0",
    "evaluation-decoder/decoder/attention_decoder/weights:0",
    "evaluation-decoder/decoder/attention_decoder/gru_cell/candidate/weights:0",
    "evaluation-decoder/decoder/attention_decoder/AttnOutputProjection/weights:0",
    "evaluation-sigmoid/weights:0",
)
