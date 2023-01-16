# Dynamic Neural Network Library with Tape-based Autodiff
## Requirements
- Python 3.8

## Setup
```sh
$ python -m pip install -r requirements.txt
$ python -m pip install -r requirements.extra.txt
$ python -m pip install -Ue .
```

## Run
```sh
$ python
>>> import simpletorch
```

## Test
```sh
$ pytest -v tests/{TARGET}.py
```

## Optimize model
```
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
for input, target in dataset:
    optimizer.zero_grad()
    output = model(input)
    loss = loss_model(output, target)
    loss.backward()
    optimizer.step()
```

## Current status (Whether the unit tests have passed)
- [x] autodiff
- [ ] conv
- [x] module
- [x] modules
- [ ] nn
- [x] operators
- [x] scalar
- [x] tensor_data
- [ ] tensor_general
- [ ] tensor
