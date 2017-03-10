WORD2VEC_EPOCHS = 500
SEQ2SEQ_EPOCHS = 2000
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
L2_WEIGHT = 0.0001

ANALYSER_REGULARIZATION_VARIABLES = (
    "analyser/head/encoder/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "analyser/head/encoder/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "analyser/param/encoder/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "analyser/param/encoder/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "analyser/variables/encoder/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "analyser/variables/encoder/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "analyser/return/encoder/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "analyser/return/encoder/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "analyser/decoder/weights:0",
    "analyser/decoder/attention_decoder/AttnW_0:0",
    "analyser/decoder/attention_decoder/weights:0",
    "analyser/decoder/attention_decoder/Attention_0/weights:0",
    "analyser/decoder/attention_decoder/AttnOutputProjection/weights:0",
    "analyser/softmax/weights:0",
)

Q_FUNCTION_REGULARIZATION_VARIABLES = (
    #     TODO:
)
