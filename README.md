### Installation
Install `scikit-motionplan` https://github.com/HiroIshida/scikit-motionplan
Install `plainmp` https://github.com/HiroIshida/plainmp
Then
```
pip install -e . -v
```

### Usage
This package is not intended to be used directly. Rather, it is intended to used in https://github.com/HiroIshida/hifuku
As long as the task definition satisfies the interface of `TaskBase` defined in rpbench/interface.py, any task is ready to be used within hifuku (implements CoverLib algorithm) package.
