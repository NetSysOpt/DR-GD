# DR-GD

This represitory is an implementation of the submission: "Solving Quadratic Programs via Deep Unrolled Douglas-Rachford Splitting"


## Dependencies
```
python=3.9.19
pytorch=2.2.2
gurobi=11.0.3
scs=3.2.4
osqp=0.6.3
```

## Experiments
To generate instances, run the codes in the datasets folder. For example, for simple(rhs), run:
```
python datasets/simple_rhs/make_dataset_perturb_gz.py
```

For training, run the following command with specification for the problem type and problem:
```
python train.py --probType simple_rhs_1000 --simpleEx 440

python train.py --probType simple --simpleVar 1000 --simpleEx 440
```

For testing, run the following command
```
python test.py --probType simple_rhs_1000 --simpleEx 100
```