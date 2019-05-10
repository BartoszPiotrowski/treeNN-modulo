# treeNN-modulo
Learning remainders of arithmetic expressions modulo n with treeNNs and
comparing it to other networks.

# Generate data
```
DATA_DIR=data/tmp/easy_very_long
mkdir -p $DATA_DIR
python3 utils/generate.py \
	-n 50000 \
	--modulo 2 \
	--numbers 0,1,2 \
	--max_length 100 \
	> $DATA_DIR/all
python3 utils/split.py $DATA_DIR/all
```

# Training
```
python3 models/tree-nn.py --data_dir $DATA_DIR --epochs 100
```

# Notes
- after doing lin/relu/lin/relu strange things started to happen
- when doing fallback to lin/relu/lin (32 units), on easy data set it works
  (accuracy = 1 after 31 epochs)
- on easy_plus_minus_minus also works (accuracy = 1 afetr 36 epochs)
- sometimes it's after 50 epochs
- with lin/relu/lin/relu training is unsuccessful
- with lin alone training is unsuccessful
- but on easy_plus and easy_plus_minus -- after 200 epochs still random
  predictions...
- when using --short_examples_first option on easy (20) acc_val == 1. in 21st epoch
- and this happens also for longer examples: easy_long (50), acc_val == 1 in 22nd epoch
- even for very long examples (100) works nicely; but look on this drop on validation:
Epoch 34. Level 5. Max length of examples 39.
( ( ( 2 - 0 ) * 1 ) - ( 0 - ( 0 - 2 ) ) ) - ( ( ( 1 * 1 ) * 0 ) + ( 0 + 1 ) )
Loss on training: 3.372462590535482e-05
Accuracy on current training subset: 1.0
Accuracy on validation: 0.9992

Epoch 35. Level 6. Max length of examples 43.
( ( 2 - 0 ) * ( 1 + 2 ) ) * ( ( ( 1 + 0 ) + 0 ) * ( ( ( 2 - 2 ) * 1 ) * ( 1 - 2 ) ) )
Loss on training: 1.8183872813270204e-05
Accuracy on current training subset: 1.0
Accuracy on validation: 1.0

Epoch 36. Level 7. Max length of examples 51.
( ( ( 0 - 0 ) + 1 ) * ( 1 - 2 ) ) * ( ( 1 - ( ( 2 * 1 ) * 0 ) ) + ( ( 2 + ( 1 + 1 ) ) * ( 2 - 0 ) ) )
Loss on training: 0.014327268466353417
Accuracy on current training subset: 0.582125
Accuracy on validation: 0.527

Epoch 37. Level 7. Max length of examples 51.
( ( ( 0 - 0 ) + 1 ) * ( 1 - 2 ) ) * ( ( 1 - ( ( 2 * 1 ) * 0 ) ) + ( ( 2 + ( 1 + 1 ) ) * ( 2 - 0 ) ) )
Loss on training: 0.0018808248999218145
Accuracy on current training subset: 1.0
Accuracy on validation: 1.0

Epoch 38. Level 8. Max length of examples 59.
( ( 0 + ( 0 - ( 1 - 0 ) ) ) - ( 0 + ( 2 + ( 0 + 1 ) ) ) ) - ( ( ( 2 + 1 ) * ( 2 + 1 ) ) * ( ( 2 + ( 2 - 2 ) ) - 1 ) )
Loss on training: 1.7237106959025066e-05
Accuracy on current training subset: 1.0
Accuracy on validation: 1.0

Epoch 39. Level 9. Max length of examples 99.
( ( ( ( ( 2 - ( 2 * 0 ) ) * ( 2 - 1 ) ) + ( 0 - 0 ) ) * 1 ) - ( ( 1 * 2 ) + ( ( ( 0 + 0 ) + 1 ) + 0 ) ) ) * ( ( ( ( 1 * 1 ) - 0 ) - ( ( 1 + ( 2 * 2 ) ) * 1 ) ) * ( 2 - ( ( 0 - 2 ) + ( 1 - 2 ) ) ) )
Loss on training: 8.108377456665039e-06
Accuracy on current training subset: 1.0
Accuracy on validation: 1.0
Trining finished.

- very strange: when lin/relu/lin/relu -- no training even on simple data


# Experiment 1
Data:
mod2_num3, mod3_num3, mod4_num3, mod5_num3
lengths upt 50
number of examples 30 000

Grid search:
learning rate = 0.01, 0.005, 0.001
short_first True, False
num_layers 1, 2
num_units 16, 32
