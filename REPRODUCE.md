# Experiments to reproduce with instructions

For all experiments, use fixed curvature values:

$c \in \{0.1, 0.5, 1.0, 2.5, 5.0\}$

Run from repository root and use the command override `model.c=<value>` in every
run.

If you want fixed curvature (not learned), also pass:
`general.learn_curvature=false`

## Map of experiment configs

- Community-small HVQVAE: `configs/experiment/comm20_hyp.yaml`
- Ego-small HVQVAE: `configs/experiment/ego_small_hyp.yaml`
- QM9 HVQVAE: `configs/experiment/qm9.yaml`
- Community-small flow matching: `configs/experiment/comm20_hyp.yaml` +
  `configs/flow_train/train.yaml`
- Ego-small flow matching: `configs/experiment/ego_small_hyp.yaml` +
  `configs/flow_train/train.yaml`
- QM9 flow matching: `configs/experiment/qm9.yaml` +
  `configs/flow_train/train.yaml`

## General run pattern

For each $c$ value, run this base command for HVQVAE pretraining:

```bash
python3 src/main.py +experiment=<experiment_name> dataset=<dataset_name>
loss=VQVAE model.c=<c_value> general.learn_curvature=false
```

For each $c$ value, run this base command for flow matching (after HVQVAE):

```bash
python3 src/train_flow.py +experiment=<experiment_name> dataset=<dataset_name> loss=VQVAE model.c=<c_value> general.learn_curvature=false flow_train.VAE_checkpoint=<path_to_vae_checkpoint>
```

## Table 1

Validation metrics: degree, clustering, orbit, average.

1. HVQVAE on Community-small
   - Config: configs/experiment/comm20_hyp.yaml
   - Command template:
     ```bash
     python3 src/main.py +experiment=comm20_hyp dataset=comm20 loss=VQVAE model.c=<c_value> general.learn_curvature=false
     ```
   - Do this for all $c$ in {0.1, 0.5, 1.0, 2.5, 5.0}.
   - Collect validation sampling metrics after training/testing from logs.

2. HVQVAE on Ego-small
   - Config: configs/experiment/ego_small_hyp.yaml
   - Command template:
     ```bash
     python3 src/main.py +experiment=ego_small_hyp dataset=ego_small loss=VQVAE model.c=<c_value> general.learn_curvature=false
     ```
   - Do this for all $c$ in {0.1, 0.5, 1.0, 2.5, 5.0}.
   - Collect validation sampling metrics after training/testing from logs.

3. HVQVAE on QM9
   - Config: configs/experiment/qm9.yaml
   - Command template:
     ```bash
     python3 src/main.py +experiment=qm9 dataset=qm9 loss=VQVAE model.c=<c_value> general.learn_curvature=false
     ```
   - Do this for all $c$ in {0.1, 0.5, 1.0, 2.5, 5.0}.
   - Collect validation sampling metrics after training/testing from logs.

## Table 2

Validation metrics: degree, clustering, orbit, average.

1. HVQVAE on Community-small
   - Use the same run as Table 1, item 1.
   - Config: configs/experiment/comm20_hyp.yaml

2. HVQVAE + flow matching on Community-small
   - HVQVAE config: configs/experiment/comm20_hyp.yaml
   - Flow config base: configs/flow_train/train.yaml
   - Step A: train HVQVAE and save best checkpoint.
   - Step B: run flow matching with that checkpoint:
     ```bash
     python3 src/train_flow.py +experiment=comm20_hyp dataset=comm20 loss=VQVAE model.c=<c_value> general.learn_curvature=false flow_train.VAE_checkpoint=<path_to_vae_checkpoint>
     ```
   - Repeat for all $c$ values.

3. HVQVAE on Ego-small
   - Use the same run as Table 1, item 2.
   - Config: configs/experiment/ego_small_hyp.yaml

4. HVQVAE + flow matching on Ego-small
   - HVQVAE config: configs/experiment/ego_small_hyp.yaml
   - Flow config base: configs/flow_train/train.yaml
   - Step A: train HVQVAE and save best checkpoint.
   - Step B: run flow matching:
     ```bash
     python3 src/train_flow.py +experiment=ego_small_hyp dataset=ego_small loss=VQVAE model.c=<c_value> general.learn_curvature=false flow_train.VAE_checkpoint=<path_to_vae_checkpoint>
     ```
   - Repeat for all $c$ values.

## Table 3

Validation metrics: valid, unique, novel, valid+novel, novel/unique,
valid+unique+novel.

1. HVQVAE on QM9
   - Config: configs/experiment/qm9.yaml
   - Command template:
     ```bash
     python3 src/main.py +experiment=qm9 dataset=qm9 loss=VQVAE model.c=<c_value> general.learn_curvature=false
     ```
   - Repeat for all $c$ values.

2. HVQVAE + flow matching on QM9
   - HVQVAE config: configs/experiment/qm9.yaml
   - Flow config base: configs/flow_train/train.yaml
   - Step A: train HVQVAE and save checkpoint.
   - Step B: run flow matching:
     ```bash
     python3 src/train_flow.py +experiment=qm9 dataset=qm9 loss=VQVAE model.c=<c_value> general.learn_curvature=false flow_train.VAE_checkpoint=<path_to_vae_checkpoint>
     ```
   - Repeat for all $c$ values.

## Figure 3

Goal: geodesic interpolation for each curvature value.

What to reproduce:

- Train the HVQVAE checkpoint for the target dataset and curvature.
- Run flow matching from that checkpoint.
- Let validation trigger the interpolation strip generation automatically.
- Compare the resulting strips across all $c$ values.

Recommended procedure:

1. Train HVQVAE with one of the experiment configs above at the target c.
2. Run flow matching with the corresponding +experiment and set
   flow_train.VAE_checkpoint to the trained HVQVAE checkpoint.
3. During flow validation, the model calls the VAE interpolation path on the
   first validation batch and writes the strip visualization under the
   interpolation output directory used by the code.
4. Repeat for all $c \in \{0.1, 0.5, 1.0, 2.5, 5.0\}$.

Where it is implemented:

- `src/HGVAE.py`: interpolation is defined in `test_interpolate` and the strip
  is written from there.
- `src/HypeFlow.py`: validation calls `self.VAE.test_interpolate(batch)`.
- Output directory: `/home/crwang/code/graph-generation/HypeFlow/data/interpolate/`
  and the graph subdirectory beneath it, for example
  `graphs/<run_name>/strips_ckpt2/...`.

### Notes:

- If you use the provided script examples in `src/script`, keep them consistent
  with these experiment names.
- Always keep dataset and +experiment matched:
  - `comm20` <-> `comm20_hyp`
  - `ego_small` <-> `ego_small_hyp`
  - `qm9` <-> `qm9`

