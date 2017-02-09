import os

RESOURCES = os.getcwd() + "/../../resources"
SAVE = RESOURCES + '/nets/javadoc->jvectors'
DATA = RESOURCES + '/dataSets/filtered.txt'
METHODS = RESOURCES + "/methods.xml"
SENTENCES = RESOURCES + "/dataSets/sentences.txt"
FILTERED = RESOURCES + "/dataSets/filtered.txt"
BATCHES = RESOURCES + "/dataSets/batches.pkl"
EPOCHS = 500
FEATURES = 100
WINDOW = 10
STATE_SIZE = 1000
BATCH_SIZE = 20
