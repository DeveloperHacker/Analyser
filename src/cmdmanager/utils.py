def unique_integer_key(dictionary: dict) -> int:
    prev = 0
    for uid in sorted(dictionary.keys()):
        if uid - prev != 1:
            return uid - 1
        prev = uid
    return prev + 1


def expand(clock: float):
    clock = int(clock * 1000)
    ms = clock % 1000
    s = clock // 1000 % 60
    m = clock // 1000 // 60 % 60
    h = clock // 1000 // 60 // 60
    if h != 0:
        return "{:02d}h {:02d}m {:02d}s {:03d}ms".format(h, m, s, ms)
    elif m != 0:
        return "{:02d}m {:02d}s {:03d}ms".format(m, s, ms)
    elif s != 0:
        return "{:02d}s {:03d}ms".format(s, ms)
    else:
        return "{:03d}ms".format(ms)
