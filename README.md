# Re.XNAS

XNAS starts from zero.

**XNAS** is an effective neural architecture search codebase, written in [PyTorch](https://pytorch.org/).

## Installation

```bash
git clone https://github.com/MAC-AutoML/XNAS.git
cd XNAS
```

## Usage

```bash
# set root path
export PYTHONPATH=$PYTHONPATH:/Path/to/XNAS
# set gpu devices
export CUDA_VISIBLE_DEVICES=0
# unit test example
python test/sng_function_optimization.py
# train example
python tools/train_darts.py --cfg configs/search/darts.yaml
# replace config example
python tools/train_darts.py --cfg configs/search/darts.yaml OUT_DIR /username/project/XNAS/experiment/darts/test1
```

## Timeline

- [x] core
- [x] datasets/utils
- [x] datasets: cifar10 & imagenet (with test case)
- [x] search_space/mb (inherit)
- [x] search_space/cellbased (with test case)
- [x] DARTS
- [x] PDARTS, PCDARTS
- [x] core: trainer & builder
- [x] merge config space
- [ ] tool: train_DARTS
- [ ] tool: train for PDARTS/PCDARTS
- [ ] search_algorithm: SNG and more (add to builder)
- [ ] tool: train_SNG
