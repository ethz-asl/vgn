# vgn

- [Data Generation](#data-generation)
- [Training](#training)
- [Evaluation](#evaluation)

## Data Generation

Collect a set of synthetic grasp attempts with

```
python scripts/generate_data.py --root path/to/dataset
```

Run `generate_data.py -h` to print the full list of arguments.
The data generation can also be distributed over multiple processors using MPI.

```
mpiexec -n <n-procs> python scripts/generate_data.py ...
```

## Training

Train a VGN model on the generated dataset using

```
python scripts/train_vgn.py --net conv --data-dir path/to/dataset --log-dir path/to/logdir
```
