# nlp_practice

This repository is dedicated to NLP tasks research and practice.

# References

1) [Stanford CS336: Language Modeling from Scratch](https://stanford-cs336.github.io/spring2025/)

How I run tests:

Invoke my code into the `stanford-cs336/assignment1-basics/tests/adapters.py` via following:

```Python
import sys

# I know exactly where I store my implementation files
# Don't do this if you store them somewhere else
module_path = os.path.abspath('../../')
sys.path.append(module_path)

from tokenization.for_tst_2 import hello_2
hello_2()
```

Next follow the official course [guide](https://github.com/stanford-cs336/assignment1-basics/tree/main?tab=readme-ov-file#setup) on how to run tests.

# 1. Used Datasets 

|ID|Name|Task|Source|
|:-|:-|:-|:-|
|1|Tatoeba|Translation|[Tatoeba Tab-delimited Bilingual Sentence Pairs](https://www.manythings.org/anki/)|
|2|TinyStories|Natural Language Corpus|[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories)|

# 2. Implemented Models



