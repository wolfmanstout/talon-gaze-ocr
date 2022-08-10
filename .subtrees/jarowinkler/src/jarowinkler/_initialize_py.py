# SPDX-License-Identifier: MIT
# Copyright (C) 2022 Max Bachmann

def _jaro_calculate_similarity(P_len: int, T_len: int, CommonChars: int, Transpositions: int) -> float:
    Transpositions //= 2
    Sim = 0.0
    Sim += CommonChars / P_len
    Sim += CommonChars / T_len
    Sim += (CommonChars - Transpositions) / CommonChars
    return Sim / 3.0

def _jaro_length_filter(P_len: int, T_len: int, score_cutoff: float) -> bool:
    """
    filter matches below score_cutoff based on string lengths
    """
    if not P_len or not T_len: return False

    sim = _jaro_calculate_similarity(P_len, T_len, min(P_len, T_len), 0)
    return sim >= score_cutoff

def _jaro_common_char_filter(P_len: int, T_len: int, CommonChars: int, score_cutoff: float) -> bool:
    """
    filter matches below score_cutoff based on string lengths and common characters
    """
    if not CommonChars: return False

    sim = _jaro_calculate_similarity(P_len, T_len, CommonChars, 0)
    return sim >= score_cutoff


def _jaro_bounds(s1, s2):
    """
    find bounds and skip out of bound parts of the sequences
    """
    P_len = len(s1)
    T_len = len(s2)

    # since jaro uses a sliding window some parts of T/P might never be in
    # range an can be removed ahead of time
    Bound = 0
    if T_len > P_len:
        Bound = T_len // 2 - 1
        if T_len > P_len + Bound:
            s2 = s2[:P_len + Bound]
    else:
        Bound = P_len // 2 - 1
        if P_len > T_len + Bound:
            s1 = s1[:T_len + Bound]
    return s1, s2, Bound

def jaro_similarity(s1, s2, *, processor=None, score_cutoff=None) -> float:
    """
    Calculates the jaro similarity

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 1.0.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 1.0

    """
    if s1 is None or s2 is None:
        return 0

    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    if score_cutoff is None:
        score_cutoff = 0

    P_len = len(s1)
    T_len = len(s2)

    # short circuit if score_cutoff can not be reached
    if not _jaro_length_filter(P_len, T_len, score_cutoff):
        return 0

    if P_len == 1 and T_len == 1:
        return float(s1[0] == s2[0])

    s1, s2, Bound = _jaro_bounds(s1, s2)

    s1_flags = [False] * P_len
    s2_flags = [False] * T_len

    # todo use bitparallel implementation
    # looking only within search range, count & flag matched pairs
    CommonChars = 0
    for i, s1_ch in enumerate(s1):
        low = max(0, i - Bound)
        hi = min(i + Bound, T_len - 1)
        for j in range(low, hi + 1):
            if not s2_flags[j] and s2[j] == s1_ch:
                s1_flags[i] = s2_flags[j] = True
                CommonChars += 1
                break

    # short circuit if score_cutoff can not be reached
    if not _jaro_common_char_filter(P_len, T_len, CommonChars, score_cutoff):
        return 0

    # todo use bitparallel implementation
    # count transpositions
    k = trans_count = 0
    for i, s1_f in enumerate(s1_flags):
        if s1_f:
            for j in range(k, T_len):
                if s2_flags[j]:
                    k = j + 1
                    break
            if s1[i] != s2[j]:
                trans_count += 1

    return _jaro_calculate_similarity(P_len, T_len, CommonChars, trans_count)
    

def jarowinkler_similarity(s1, s2, *, prefix_weight=0.1, processor=None, score_cutoff=None) -> float:
    """
    Calculates the jaro winkler similarity

    Parameters
    ----------
    s1 : Sequence[Hashable]
        First string to compare.
    s2 : Sequence[Hashable]
        Second string to compare.
    prefix_weight : float, optional
        Weight used for the common prefix of the two strings.
        Has to be between 0 and 0.25. Default is 0.1.
    processor: callable, optional
        Optional callable that is used to preprocess the strings before
        comparing them. Default is None, which deactivates this behaviour.
    score_cutoff : float, optional
        Optional argument for a score threshold as a float between 0 and 1.0.
        For ratio < score_cutoff 0 is returned instead. Default is 0,
        which deactivates this behaviour.

    Returns
    -------
    similarity : float
        similarity between s1 and s2 as a float between 0 and 1.0

    Raises
    ------
    ValueError
        If prefix_weight is invalid
    """
    if s1 is None or s2 is None:
        return 0

    if processor is not None:
        s1 = processor(s1)
        s2 = processor(s2)

    if score_cutoff is None:
        score_cutoff = 0

    P_len = len(s1)
    T_len = len(s2)
    min_len = min(P_len, T_len)
    prefix = 0
    max_prefix = min(min_len, 4)

    for _ in range(max_prefix):
        if s1[prefix] != s2[prefix]:
            break
        prefix += 1

    jaro_score_cutoff = score_cutoff
    if (jaro_score_cutoff > 0.7):
        prefix_sim = prefix * prefix_weight

        if (prefix_sim >= 1.0):
            jaro_score_cutoff = 0.7
        else:
            jaro_score_cutoff = max(0.7, (prefix_sim - jaro_score_cutoff) / (prefix_sim - 1.0))

    Sim = jaro_similarity(s1, s2, score_cutoff=jaro_score_cutoff)
    if (Sim > 0.7):
        Sim += prefix * prefix_weight * (1.0 - Sim)

    return Sim if Sim >= score_cutoff else 0
