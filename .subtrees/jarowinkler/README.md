
<h1 align="center">
 JaroWinkler
</h1>
<p align="center">
  <a href="https://github.com/maxbachmann/JaroWinkler/actions">
    <img src="https://github.com/maxbachmann/JaroWinkler/workflows/Build/badge.svg"
         alt="Continous Integration">
  </a>
  <a href="https://pypi.org/project/jarowinkler/">
    <img src="https://img.shields.io/pypi/v/jarowinkler"
         alt="PyPI package version">
  </a>
  <a href="https://www.python.org">
    <img src="https://img.shields.io/pypi/pyversions/jarowinkler"
         alt="Python versions">
  </a><br/>
  <a href="https://github.com/maxbachmann/JaroWinkler/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/maxbachmann/JaroWinkler"
         alt="GitHub license">
  </a>
</p>

<h4 align="center">JaroWinkler is a library to calculate the Jaro and Jaro-Winkler similarity. It is easy to use, is far more performant than all alternatives and is designed to integrate seemingless with <a href="https://github.com/maxbachmann/RapidFuzz">RapidFuzz</a>.</h4>



## :zap: Quickstart
```python
>>> from jarowinkler import *

>>> jaro_similarity("Johnathan", "Jonathan")
0.8796296296296297

>>> jarowinkler_similarity("Johnathan", "Jonathan")
0.9037037037037037
```

## 🚀 Benchmarks
The implementation is based on a novel approach to calculate the Jaro-Winkler similarity using bitparallelism. This is significantly faster than the original approach used in other libraries. The following benchmark shows the performance difference to jellyfish and python-Levenshtein. 

<p align="center">
<img src="https://raw.githubusercontent.com/maxbachmann/JaroWinkler/main/bench/results/JaroWinkler.svg?sanitize=true" alt="Benchmark JaroWinkler">
</p>

## ⚙️ Installation

You can install this library from [PyPI](https://pypi.org/project/jarowinkler/) with pip:
```
pip install jarowinkler
```
JaroWinkler provides binary wheels for all common platforms.

### Source builds

For a source build (for example from a SDist packaged) you only require a C++14 compatible compiler. You can install directly from GitHub if you would like.
```
pip install git+https://github.com/maxbachmann/JaroWinkler.git@main
```

## 📖 Usage

Any algorithms in JaroWinkler can not only be used with strings, but with any arbitary sequences of hashable objects:
```python
from jarowinkler import jarowinkler_similarity


jarowinkler_similarity("this is an example".split(), ["this", "is", "a", "example"])
# 0.8666666666666667
```

So as long as two objects have the same hash they are treated as similar. You can provide a `__hash__` method for your own object instances.

```python
class MyObject:
    def __init__(self, hash):
        self.hash = hash

    def __hash__(self):
        return self.hash

jarowinkler_similarity([MyObject(1), MyObject(2)], [MyObject(1), MyObject(2), MyObject(3)])
# 0.9111111111111111
```

All algorithms provide a `score_cutoff` parameter. This parameter can be used to filter out bad matches. Internally this allows JaroWinkler to select faster implementations in some places:

```python
jaro_similarity("Johnathan", "Jonathan", score_cutoff=0.9)
# 0.0

jaro_similarity("Johnathan", "Jonathan", score_cutoff=0.85)
# 0.8796296296296297
```

JaroWinkler can be used with RapidFuzz, which provides multiple methods to compute string metrics on collections of inputs. JaroWinkler implements the RapidFuzz C-API which allows RapidFuzz to call the functions without any of the usual overhead of python, which makes this even faster.

```python
from rapidfuzz import process

process.cdist(["Johnathan", "Jonathan"], ["Johnathan", "Jonathan"], scorer=jarowinkler_similarity)
array([[1.       , 0.9037037],
       [0.9037037, 1.       ]], dtype=float32)
```

## 👍 Contributing

PRs are welcome!
- Found a bug? Report it in form of an [issue](https://github.com/maxbachmann/JaroWinkler/issues) or even better fix it!
- Can make something faster? Great! Just avoid external dependencies and remember that existing functionality should still work.
- Something else that do you think is good? Do it! Just make sure that CI passes and everything from the README is still applicable (interface, features, and so on).
- Have no time to code? Tell your friends and subscribers about JaroWinkler. More users, more contributions, more amazing features.

Thank you :heart:

## ⚠️ License
Copyright 2021 - present [maxbachmann](https://github.com/maxbachmann). `JaroWinkler` is free and open-source software licensed under the [MIT License](https://github.com/maxbachmann/JaroWinkler/blob/main/LICENSE).
