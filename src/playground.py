from utils import Score

p = 7
a = [[-1, -1, -1, -1], [1, 2, p, p], [-1, -1, -1, -1], [-1, -1, -1, -1], [1, 3, 6, p], [-1, -1, -1, -1]]
b = [[+1, +5, +9, +8], [8, 2, p, 1], [+1, +2, +4, +2], [+3, +1, +2, +1], [1, 1, p, 2], [+2, +1, +p, +2]]
sc = Score.calc(a, b, -1, p)
print("ACC", sc.accuracy)
print("F1", sc.F_score(1))
print("J", sc.jaccard)
print("TP", sc.true_positive)
print("TN", sc.true_negative)
print("FP", sc.false_positive)
print("FN", sc.false_negative)
