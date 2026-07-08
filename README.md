# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/btvn-quan-le-thien/spatialx-transform/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                                |    Stmts |     Miss |   Cover |   Missing |
|---------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/spatialx\_transform/\_\_init\_\_.py             |       10 |        1 |     90% |        50 |
| src/spatialx\_transform/\_version.py                |       11 |        0 |    100% |           |
| src/spatialx\_transform/params/\_\_init\_\_.py      |        5 |        0 |    100% |           |
| src/spatialx\_transform/params/affine\_params.py    |       12 |        0 |    100% |           |
| src/spatialx\_transform/params/composed\_params.py  |        7 |        0 |    100% |           |
| src/spatialx\_transform/params/tps\_params.py       |       12 |        0 |    100% |           |
| src/spatialx\_transform/params/transform\_params.py |       10 |        1 |     90% |        13 |
| src/spatialx\_transform/point.py                    |       59 |        2 |     97% |    28, 36 |
| src/spatialx\_transform/transforms/\_\_init\_\_.py  |        7 |        0 |    100% |           |
| src/spatialx\_transform/transforms/affine.py        |       27 |        0 |    100% |           |
| src/spatialx\_transform/transforms/composed.py      |       15 |        0 |    100% |           |
| src/spatialx\_transform/transforms/identity.py      |       10 |        0 |    100% |           |
| src/spatialx\_transform/transforms/square.py        |       12 |        0 |    100% |           |
| src/spatialx\_transform/transforms/tps.py           |       83 |       13 |     84% |47, 84, 104-118 |
| src/spatialx\_transform/transforms/transform.py     |       58 |        2 |     97% |    62, 66 |
| src/spatialx\_transform/warp.py                     |      191 |       10 |     95% |101, 109, 214, 248, 277, 289-290, 412-416 |
| **TOTAL**                                           |  **529** |   **29** | **95%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/btvn-quan-le-thien/spatialx-transform/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/btvn-quan-le-thien/spatialx-transform/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/btvn-quan-le-thien/spatialx-transform/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/btvn-quan-le-thien/spatialx-transform/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fbtvn-quan-le-thien%2Fspatialx-transform%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/btvn-quan-le-thien/spatialx-transform/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.