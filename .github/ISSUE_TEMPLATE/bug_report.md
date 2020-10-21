---
name: Bug report
about: Create a report to help us improve
title: ''
labels: bug
assignees: ''



**Describe the bug**
A clear and concise description of what the bug is.

**To Reproduce**
Steps to reproduce the behavior:
1. Go to '...'
2. Use data '...'
3. Run '....' with arguments '...'
4. See error

**Expected behavior**
A clear and concise description of what you expected to happen.

**Screenshots**
If applicable, add screenshots to help explain your problem.

**Environment (please complete the following information):**
 - OS: [e.g. Linux]
 - OpenKiwi version [e.g. 0.1.0]
 - Python version

**Additional context**
Add any other context about the problem here.


---


**Describe the bug**
Searching for related issues found that when the `InputFields` class in `kiwi/data/encoders/wmt_qe_data_encoder.py` was defined, `GenericMode` and `Generic[T]` conflict caused

**To Reproduce**
Steps to reproduce the behavior:
1. Creat a new file: run.py
```
from kiwi.lib.train import train_from_file
run_info = train_from_file('config/bert.yaml')
```
and then execute this program
2. See error
TypeError: metaclass conflict: the metaclass of a derived class must be a (non-strict) subclass of the metaclasses of all its bases

**Expected behavior**
I hope to pass the type check when the parameter is initialized

**Screenshots**
https://github.com/Shannen3206/csn/blob/master/error.png

**Environment (please complete the following information):**
 - OS: Linux
 - OpenKiwi version： 2.0.0
 - Python version：3.6.9

---
