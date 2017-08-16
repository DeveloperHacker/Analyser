from typing import List

import numpy as np

EPSILON = 1e-10


class Score:
    def __init__(self, TP, TN, FP, FN):
        self.TP = TP
        self.TN = TN
        self.FP = FP
        self.FN = FN
        self.Ps = self.TP + self.FP
        self.Ns = self.TN + self.FN
        self.T = self.TP + self.TN
        self.F = self.FP + self.FN
        self.P = self.TP + self.FN
        self.N = self.TN + self.FP
        self.ALL = self.P + self.N
        self.TPR = self.TP / (self.P + EPSILON)
        self.TNR = self.TN / (self.N + EPSILON)
        self.PPV = self.TP / (self.Ps + EPSILON)
        self.NPV = self.TN / (self.Ns + EPSILON)
        self.FNR = 1 - self.TPR
        self.FPR = 1 - self.TNR
        self.FDR = 1 - self.PPV
        self.FOR = 1 - self.NPV
        self.ACC = self.T / self.ALL
        self.MCC = (self.TP * self.TN - self.FP * self.FN) / (np.sqrt(self.Ps * self.P * self.N * self.Ns) + EPSILON)
        self.BM = self.TPR + self.TNR - 1
        self.MK = self.PPV + self.NPV - 1
        self.J = self.TP / (self.TP + self.FP + self.FN + EPSILON)
        self.recall = self.TPR
        self.precision = self.PPV
        self.accuracy = self.ACC
        self.true_positive = self.TP
        self.true_negative = self.TN
        self.false_negative = self.FN
        self.false_positive = self.FP
        self.jaccard = self.J

    def F_score(self, beta):
        beta = beta * beta
        return (1 + beta) * self.PPV * self.TPR / (beta * self.PPV + self.TPR + EPSILON)

    def E_score(self, alpha):
        return 1 - self.PPV * self.TPR / (alpha * self.TPR + (1 - alpha) * self.PPV + EPSILON)


class BatchScore:
    def __init__(self, scores: List[Score]):
        self.scores = scores
        self.TP = np.mean([score.TP / score.ALL for score in scores])
        self.TN = np.mean([score.TN / score.ALL for score in scores])
        self.FP = np.mean([score.FP / score.ALL for score in scores])
        self.FN = np.mean([score.FN / score.ALL for score in scores])
        self.Ps = self.TP + self.FP
        self.Ns = self.TN + self.FN
        self.T = self.TP + self.TN
        self.F = self.FP + self.FN
        self.P = self.TP + self.FN
        self.N = self.TN + self.FP
        self.ALL = self.P + self.N
        self.TPR = np.mean([score.TPR for score in scores])
        self.TNR = np.mean([score.TNR for score in scores])
        self.PPV = np.mean([score.PPV for score in scores])
        self.NPV = np.mean([score.NPV for score in scores])
        self.FNR = np.mean([score.FNR for score in scores])
        self.FPR = np.mean([score.FPR for score in scores])
        self.FDR = np.mean([score.FDR for score in scores])
        self.FOR = np.mean([score.FOR for score in scores])
        self.ACC = np.mean([score.ACC for score in scores])
        self.MCC = np.mean([score.MCC for score in scores])
        self.BM = np.mean([score.BM for score in scores])
        self.MK = np.mean([score.MK for score in scores])
        self.recall = self.TPR
        self.precision = self.PPV
        self.accuracy = self.ACC
        self.true_positive = self.TP
        self.true_negative = self.TN
        self.false_negative = self.FN
        self.false_positive = self.FP

    def F_score(self, beta):
        return np.mean([score.F_score(beta) for score in self.scores])

    def E_score(self, alpha):
        return np.mean([score.E_score(alpha) for score in self.scores])