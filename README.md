[![Build Status](https://app.travis-ci.com/qcwang77/sbs_mxnet2keras.svg?branch=master)](https://app.travis-ci.com/qcwang77/sbs_mxnet2keras)[![Coverage Status](https://coveralls.io/repos/github/qcwang77/sbs_mxnet2keras/badge.svg?branch=master)](https://coveralls.io/github/qcwang77/sbs_mxnet2keras?branch=master)


# sbs-mxnet2keras
Deep learning model conversion and visualization tool from Mxnet framework to Keras framework.

## Project Organization
```
sbs-mxnet2keras/
  |- README.md
  |- mxnet2keras/
    |- __init__.py
    |- Model_Summary.py
    |- Weight_Converter.py
    |- Weight_Transfer_Functions.py
    |- tests/
      |- __init__.py
      |- test_Weight_Converter.py
      |- test_Model_Summary.py
      |- build_net_for_test.py
  |- data/
    |- cnocr-v1.2.0-conv-lite-fc-0025.params
    |- cnocr-v1.2.0-conv-lite-fc-symbol.json
  |-example/
    |-example.ipynb
  |- setup.py
  |- requirements.txt
  |- LICENSE
```
---
## Dependencies

To check the dependencies of this package,
refer to the [requirements](/requirements.txt)

## Installation

Clone the repo and create a virtual environment in the root of the repo
```bash
python -m venv venv
source venv/bin/activate
```

Install the dependencies from the `requirements.txt` file using
```bash
python -m pip install -r requirements.txt
```

Install the package using the following command
```bash
python setup.py sdist
```

This will generate the pip installation package `mxnet2keras-0.0.1.tar.gz` in the `dist/` directory.
The package `mxnet2keras` can now be installed using

```bash
pipi nstall ./dist/mxnet2keras-0.0.1.tar.gz
```

## Usage

To see how to use the package to visualize model and convert model weight, 
refer to the [example notebook](mxnet2keras/example/example.ipynb)