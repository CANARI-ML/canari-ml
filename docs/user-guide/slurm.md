# SLURM Submission

Running jobs on clusters is relatively straightforward. 

## Preprocessing

As an example, for preprocessing:

``` console
canari_ml preprocess train 'hydra.searchpath=[file://.]' hydra/launcher=slurm_jasmin_cpu -cd configs/preprocess/ -cn train_demo_dataset -m
```

- The `-m` flag is used to submit the job to SLURM.
- `'hydra.searchpath=[file://.]'` is used to define where additional config files should be found. This lets you define config paths relative to your current location. In this case, `hydra/launcher=slurm_jasmin_cpu.yaml` .

where, this custom config file is used to define the job submission on a SLURM cluster. As an example, for the [JASMIN LOTUS2 cluster](https://help.jasmin.ac.uk/docs/batch-computing/lotus-overview/), you would use the following config file:

``` yaml title="hydra/launcher/slurm_jasmin_cpu.yaml"
# @package hydra.launcher
_target_: hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher

submitit_folder: ${hydra.sweep.dir}/${hydra.sweep.subdir}/slurm/%j
timeout_min: 1440 #(1)!
cpus_per_task: 1
tasks_per_node: 1
mem_gb: 256
nodes: 1
name: ${hydra.job.name}
partition: standard
qos: high
comment: null
mem_per_cpu: null
account: canari
max_num_timeout: 0
setup:   #(2)!
   - echo "Hello World"
```

1. Duration to run for in minutes, in this case, 24 hours.
2. List of set-up commands to run before the job starts.

## Training

And, for submitting a training run on a GPU partition:

``` console
canari_ml train 'hydra.searchpath=[file://.]' hydra/launcher=slurm_jasmin_cpu train.dataset=preprocessed_data/train_demo_dataset/03_cache_demo_dataset/cached.DAY.north.json train.name=demo_train train.epochs=2
```

``` yaml title="hydra/launcher/slurm_jasmin_gpu.yaml"
# @package hydra.launcher
_target_: hydra_plugins.hydra_submitit_launcher.submitit_launcher.SlurmLauncher

submitit_folder: ${hydra.sweep.dir}/${hydra.sweep.subdir}/slurm/%j
timeout_min: 1440
cpus_per_task: 4
tasks_per_node: 1
mem_gb: 256
nodes: 1
name: ${hydra.job.name}
partition: orchid
qos: orchid
comment: null
gres: "gpu:1"  # (1)!
mem_per_cpu: null
account: canari
max_num_timeout: 0
setup:
   - echo "Hello World"
```

1. This defines how many GPU resources you are requesting for training.
