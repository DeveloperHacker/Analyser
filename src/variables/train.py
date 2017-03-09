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
L2_WEIGHT = 0.0001

REGULARIZATION_VARIABLES = (
    "results/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "results/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "params/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "params/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "variables/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "variables/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "head/bidirectional_rnn/fw/gru_cell/candidate/weights:0",
    "head/bidirectional_rnn/bw/gru_cell/candidate/weights:0",
    "attention_decoder/output/weights:0",
    "attention_decoder/gru_cell/candidate/weights:0",
    "attention_decoder/AttnW_0:0",
    "attention_decoder/Attention_0/weights:0",
    "attention_decoder/AttnOutputProjection/weights:0"
)
