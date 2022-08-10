from string import ascii_letters, digits, punctuation

from hypothesis import given, settings
import hypothesis.strategies as st
from jarowinkler import _initialize_py, _initialize_cpp

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def jarowinkler_similarity(*args, **kwargs):
    sim1 = _initialize_py.jarowinkler_similarity(*args, **kwargs)
    sim2 = _initialize_cpp.jarowinkler_similarity(*args, **kwargs)
    assert isclose(sim1, sim2)
    return sim1

def jaro_similarity(P, T):
    P_flag = [0] * (len(P) + 1)
    T_flag = [0] * (len(T) + 1)

    Bound = max(len(P), len(T)) // 2
    Bound = max(Bound - 1, 0)

    CommonChars = 0
    for i in range(len(T)):
        lowlim = i - Bound if i >= Bound else 0
        hilim = i + Bound if i + Bound <= len(P) - 1 else len(P) - 1

        for j in range(lowlim, hilim + 1):
            if not P_flag[j] and P[j] == T[i]:
                T_flag[i] = 1
                P_flag[j] = 1
                CommonChars += 1
                break

    if not CommonChars:
        return 0

    Transpositions = 0
    k = 0
    for i in range(len(T)):
        if T_flag[i]:
            j = k
            while j < len(P):
                if P_flag[j]:
                    k = j + 1
                    break
                j += 1

            if T[i] != P[j]:
                Transpositions += 1

    Transpositions = Transpositions // 2

    Sim = CommonChars / len(P) + CommonChars / len(T) + (CommonChars - Transpositions) / CommonChars
    return Sim / 3

def jaro_winkler_similarity(P, T, prefix_weight=0.1):
    min_len = min(len(P), len(T))
    prefix = 0
    max_prefix = min(min_len, 4)

    while prefix < max_prefix:
        if T[prefix] != P[prefix]:
            break
        prefix += 1

    Sim = jaro_similarity(P, T)
    if Sim > 0.7:
        Sim += prefix * prefix_weight * (1.0 - Sim)

    return Sim


def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


HYPOTHESIS_ALPHABET = ascii_letters + digits + punctuation

@given(s1=st.text(max_size=64), s2=st.text(max_size=64))
@settings(max_examples=50, deadline=1000)
def test_jaro_winkler_word(s1, s2):
    assert isclose(jaro_winkler_similarity(s1, s2), jarowinkler_similarity(s1, s2))

@given(s1=st.text(min_size=65), s2=st.text(min_size=65))
@settings(max_examples=50, deadline=1000)
def test_jaro_winkler_block(s1, s2):
    assert isclose(jaro_winkler_similarity(s1, s2), jarowinkler_similarity(s1, s2))

@given(s1=st.text(), s2=st.text())
@settings(max_examples=50, deadline=1000)
def test_jaro_winkler_random(s1, s2):
    print(s1, s2)
    assert isclose(jaro_winkler_similarity(s1, s2), jarowinkler_similarity(s1, s2))
