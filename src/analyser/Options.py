class Options:
    def __init__(self):
        self.epochs = None
        self.batch_size = None
        self.inputs_state_size = None
        self.inputs_hidden_size = None
        self.labels_state_size = None
        self.labels_hidden_size = None
        self.tokens_state_size = None
        self.tokens_hidden_size = None
        self.strings_state_size = None
        self.strings_hidden_size = None
        self.l2_weight = None
        self.minimum_length = None
        self.train_set = None
        self.validation_set = None
        self.test_set = None
        self.model_dir = None
        self.data_set_path = None
        self.summaries_dir = None
        self.tokens_output_type = None
        self.flatten_type = None

    def validate(self):
        assert self.epochs is not None
        assert self.batch_size is not None
        assert self.inputs_state_size is not None
        assert self.labels_state_size is not None
        assert self.tokens_state_size is not None
        assert self.strings_state_size is not None
        assert self.l2_weight is not None
        assert self.minimum_length is not None
        assert self.train_set is not None
        assert self.validation_set is not None
        assert self.test_set is not None
        assert self.model_dir is not None
        assert self.data_set_path is not None
        assert self.summaries_dir is not None
        assert self.tokens_output_type is not None
        assert self.tokens_output_type in ("tree", "sequence")
        assert self.flatten_type is not None
        assert self.flatten_type in ("dfs", "bfs")
        assert self.tokens_output_type != "tree" or self.flatten_type == "bfs"

    @staticmethod
    def value_of(json_object: dict) -> 'Options':
        options = Options()
        options.epochs = json_object.get("epochs", None)
        options.batch_size = json_object.get("batch_size", None)
        options.inputs_state_size = json_object.get("inputs_state_size", None)
        options.inputs_hidden_size = json_object.get("inputs_hidden_size", None)
        options.labels_state_size = json_object.get("labels_state_size", None)
        options.labels_hidden_size = json_object.get("labels_hidden_size", None)
        options.tokens_state_size = json_object.get("tokens_state_size", None)
        options.tokens_hidden_size = json_object.get("tokens_hidden_size", None)
        options.strings_state_size = json_object.get("strings_state_size", None)
        options.strings_hidden_size = json_object.get("strings_hidden_size", None)
        options.l2_weight = json_object.get("l2_weight", None)
        options.minimum_length = json_object.get("minimum_length", None)
        options.train_set = json_object.get("train_set", None)
        options.validation_set = json_object.get("validation_set", None)
        options.test_set = json_object.get("test_set", None)
        options.model_dir = json_object.get("model_dir", None)
        options.data_set_path = json_object.get("data_set_path", None)
        options.summaries_dir = json_object.get("summaries_dir", None)
        options.tokens_output_type = json_object.get("tokens_output_type", None)
        options.flatten_type = json_object.get("flatten_type", None)
        return options

    def serialize(self) -> dict:
        json_object = {
            "epochs": int(self.epochs),
            "batch_size": int(self.batch_size),
            "inputs_state_size": int(self.inputs_state_size),
            "inputs_hidden_size": self.inputs_hidden_size,
            "labels_state_size": int(self.labels_state_size),
            "labels_hidden_size": self.labels_hidden_size,
            "tokens_state_size": int(self.tokens_state_size),
            "tokens_hidden_size": self.tokens_hidden_size,
            "strings_state_size": int(self.strings_state_size),
            "strings_hidden_size": self.strings_hidden_size,
            "l2_weight": float(self.l2_weight),
            "minimum_length": int(self.minimum_length),
            "train_set": float(self.train_set),
            "validation_set": float(self.validation_set),
            "test_set": float(self.test_set),
            "model_dir": str(self.model_dir),
            "data_set_path": str(self.data_set_path),
            "summaries_dir": str(self.summaries_dir),
            "tokens_output_type": str(self.tokens_output_type),
            "flatten_type": str(self.flatten_type)}
        return json_object
