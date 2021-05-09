def compute_f_score(tp, fn, fp):
    r"""
    :param tp: int, true positive
    :param fn: int, false negative
    :param fp: int, false positive
    :return: (f, pre, rec)
    """
    epsilon = 1e-13
    pre = tp / (fp + tp + epsilon)
    rec = tp / (fn + tp + epsilon)
    f = 2 * pre * rec / (pre + rec + 1e-13)

    return f, pre, rec