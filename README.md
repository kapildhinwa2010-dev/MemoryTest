# MemoryTest

MemoryTest is a small Python/NumPy implementation of a bipolar associative-memory experiment. It stores patterns in a Hebbian outer-product matrix and provides correlation-based recall using stored pattern copies, plus iterative sign-update recall from the matrix alone. It is an experimental, minimal library rather than a validated high-capacity memory system; the included tests check basic behavior and input validation, not general recall reliability or performance.

## Features

- Stores one-dimensional patterns whose entries are `-1` or `+1` in a `dim × dim` floating-point lattice.
- Supports writing one pattern at a time (`write`) or an iterable (`write_many`).
- `recall` returns the stored pattern with the highest dot-product correlation to a cue. This mode requires the stored pattern copies to remain available.
- `recall_from_lattice` performs a configurable number of synchronous sign updates using the accumulated lattice and can be used after `forget_copies()` clears the stored copies.
- Provides `hamming_distance`, a cleanup helper (`cleanup`), and a random bipolar-pattern helper (`noise_pattern`).

## Requirements and installation

Python 3.9 or newer and NumPy 1.24 or newer are declared runtime requirements. Install the project from its repository root with:

```bash
python -m pip install .
```

To install the dependencies used by the test suite as well:

```bash
python -m pip install -r requirements.txt
```

The project configuration declares the `memorytest` package at version `0.1.0`. Wheel building was verified from the inspected source snapshot.

## Basic usage

```python
import numpy as np
from memorytest import WaveMemory

rng = np.random.default_rng(0)
pattern = rng.choice([-1, 1], size=64)

memory = WaveMemory(dim=64)
memory.write(pattern)

cue = pattern.copy()
cue[[2, 11, 25]] *= -1
recovered = memory.recall(cue)
print(np.array_equal(recovered, pattern))
```

`recall` compares the cue against the stored copies and selects the largest dot product. It is not a lattice-only operation. To discard stored copies and use the lattice-only method instead:

```python
memory.forget_copies()
recovered = memory.recall_from_lattice(cue, steps=3)
```

The lattice method applies `sign(lattice @ state)` repeatedly. Its output depends on the stored patterns, cue corruption, and number of steps; this project does not establish a general noise tolerance or capacity guarantee.

## Helpers and limitations

- `hamming_distance(a, b)` returns the number of unequal elements and raises `ValueError` if the shapes differ.
- `cleanup(noisy, max_flips=8)` uses the copy-based `recall` method and raises `ValueError` when its computed difference exceeds the configured percentage threshold. It therefore requires stored copies and should not be treated as a general noise-robustness guarantee.
- `noise_pattern(shape, flip_prob, rng)` currently returns a newly sampled random `-1`/`+1` array. Although it validates that `flip_prob` is between zero and one, the implementation does not use that value to control the output and does not flip a supplied base pattern. Do not use it as a calibrated corruption helper.
- The lattice uses `O(dim²)` storage, in addition to the retained pattern copies. The repository contains no benchmark results establishing performance, storage capacity, or robust operation at scale.

## Tests and CI

Run the tests with:

```bash
python -m pytest -q
```

On the inspected commit (`2c1b511455dac6cb49e59c5585472195091651c9`), the test suite was run locally and **15 tests passed**. This is a local result for that checkout, not a claim about a separate hosted CI run. The GitHub Actions CI workflow configures tests, compilation, and a small import smoke test on Python 3.10, 3.11, and 3.12. The project metadata declares Python 3.9 or newer; the CI matrix does not test 3.9.

## License

MIT License. See [LICENSE](LICENSE) for the full terms and copyright notice.
