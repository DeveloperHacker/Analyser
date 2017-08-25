import numpy as np

from analyser.misc import greedy_correct, batch_greedy_correct

targets = [[
    [[1, 2, 3], [0, 0, 1], [2, 8, 7]],
    [[1, 2, 3], [0, 0, 1], [2, 8, 7]],
    [[1, 2, 3], [0, 0, 1], [2, 8, 7]]
]]
outputs = [[
    [[3, 4, 5], [1, 2, 3], [1, 0, 2]],
    [[3, 4, 5], [1, 2, 3], [1, 0, 2]],
    [[3, 4, 5], [1, 2, 3], [1, 0, 2]]
]]
dependencies = [[
    [[1, 2, 3], [0, 0, 0], [0, 2, 3]],
    [[0, 5, 6], [3, 6, 4], [4, 5, 3]],
    [[1, 8, 3], [1, 4, 5], [4, 5, 5]]
]]
targets, dependencies = batch_greedy_correct(targets, outputs, dependencies)
print("targets")
print(targets)
print("dependencies")
print(dependencies)
