RESOURCES = "resources"
DATA_SETS = RESOURCES + "/data-sets"
FULL_DATA_SET = DATA_SETS + "/data-set.json"
JODA_TIME_DATA_SET = DATA_SETS + "/joda-time.json"

# Generator
GENERATOR = RESOURCES + "/generator"
EMBEDDINGS = GENERATOR + "/embeddings.pickle"
FILTERED = GENERATOR + "/filtered.txt"
GENERATOR_MODEL = GENERATOR + "/model.ckpt"
GENERATOR_LOG = GENERATOR + "/generator.log"

# Analyser
ANALYSER = RESOURCES + "/analyser"
ANALYSER_SUMMARIES = ANALYSER + "/summaries"
ANALYSER_RAW_DATA_SET = JODA_TIME_DATA_SET
ANALYSER_DATA_SET = ANALYSER + "/data-set.pickle"
