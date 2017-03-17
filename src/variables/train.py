WORD2VEC_EPOCHS = 500
CONTRACT_EPOCHS = 100
ANALYSER_EPOCHS = 2000 // CONTRACT_EPOCHS
Q_FUNCTION_EPOCHS = 2000 // CONTRACT_EPOCHS

EMBEDDING_SIZE = 100
WINDOW = 7
INPUT_STATE_SIZE = 200
OUTPUT_STATE_SIZE = 200
BATCH_SIZE = 20
INPUT_SIZE = 30
OUTPUT_SIZE = 10

CONTAINS_WEIGHT = 1
VARIANCE_WEIGHT = 1
Q_WEIGHT = 1
DIFF_WEIGHT = 1
L2_WEIGHT = 0.0001

REGULARIZATION_VARIABLES = (
    "head/encoder/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "head/encoder/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "param/encoder/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "param/encoder/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "variable/encoder/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "variable/encoder/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "return/encoder/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "return/encoder/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "decoder/weights:0",
    "decoder/attention_decoder/AttnW_0:0",
    "decoder/attention_decoder/weights:0",
    "decoder/attention_decoder/Attention_0/weights:0",
    "decoder/attention_decoder/AttnOutputProjection/weights:0",
    "softmax/weights:0",
)
