WORD2VEC_EPOCHS = 500
MUNCHHAUSEN_PRETRAIN_EPOCHS = 30
MUNCHHAUSEN_TRAIN_EPOCHS = 50
MUNCHHAUSEN_RUNS = 2000

TRAIN_SET = 0.9
VALIDATION_SET = 0.1

EMBEDDING_SIZE = 100
WINDOW = 7
INPUT_STATE_SIZE = 150
OUTPUT_STATE_SIZE = 150
BATCH_SIZE = 20
INPUT_SIZE = 30
OUTPUT_SIZE = 10
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
