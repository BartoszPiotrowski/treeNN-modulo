# treeNN-modulo
Learning remainders of arithmetic expressions modulo n with treeNNs and
comparing it to other networks.

# Generate data
```
DATA_DIR=data/tmp/try3
mkdir $DATA_DIR
python3 utils/generate.python3 -n=10000 --max_length=40 > $DATA_DIR/all
python3 utils/split.python3 $DATA_DIR/all
```

# Training
```
python3 models/tree-nn.python3 --data_dir data/tmp/try3 --epochs 200
```
