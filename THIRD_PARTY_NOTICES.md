# Third-party notices

QueryLink's original implementation is licensed under the Apache License 2.0. The following portions are derived from third-party software and retain the corresponding upstream copyright and license notices.

## A-Mem

Portions of the following files are adapted from [A-Mem](https://github.com/WujiangXu/A-mem):

- `src/components/amem/memory_layer.py`
- `src/components/evaluation.py`

Upstream license checked at commit `0c8039f28fdcc08189a23c07a3437d9d2482f9c2`.

MIT License

Copyright (c) 2025 Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

The A-Mem-derived code has been modified to integrate with QueryLink's configuration, memory, and task abstractions.

## Mem0

The answer-judging prompt in `src/components/utils.py` is adapted from the evaluation methodology described by [Mem0](https://github.com/mem0ai/mem0). Mem0 is licensed under the Apache License 2.0; the full license terms are reproduced in this repository's `LICENSE` file.

Upstream license checked at commit `965140eb190bf390979d41bb6ffff12c0b02e70b`. The prompt is retained to preserve comparability with the paper's reported evaluation.
