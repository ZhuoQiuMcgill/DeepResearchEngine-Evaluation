# Notice

The rubrics in `rubrics/` are taken unchanged from **ResearchRubrics: A Benchmark of
Prompts and Rubrics for Evaluating Deep Research Agents** (Sharma et al., 2025;
[arXiv:2511.07685](https://arxiv.org/abs/2511.07685); data:
[ScaleAI/researchrubrics](https://huggingface.co/datasets/ScaleAI/researchrubrics); code:
[scaleapi/researchrubrics](https://github.com/scaleapi/researchrubrics)). The grading
instructions in `RULES.md` adapt the benchmark's released grading prompt from two levels
to three. For the judging itself only the criterion texts, axes and weights were used; the
request texts were not part of what the judges saw. This branch adds the six request texts
and their labels, unchanged, in `analysis/explorer/questions.json` for the results page.

The benchmark is released under the MIT License, reproduced as published:

```text
MIT License

Copyright (c) 2025 [Authors/Institution]

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
```
