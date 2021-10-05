[Dataset](docs/crello-dataset.md) | [arXiv](https://arxiv.org/abs/2108.01249)

# CanvasVAE

Official tensorflow implementation of the following work.

> Kota Yamaguchi, CanvasVAE: Learning to Generate Vector Graphic Documents, ICCV 2021

![Interpolation](docs/interpolation.svg)

## Content

- `bin`: Job launchers
- `src/preprocess`: Preprocessing jobs to fetch and build TFRecord dataset
- `src/pixel-vae`: PixelVAE trainer
- `src/canvas-vae`: CanvasVAE trainer and evaluation

## Setup

Install python dependencies. Perhaps this should be done inside `venv`.

```bash
pip install -r requirements.txt
```

Note that Tensorflow has a version-specific system requirement for GPU environment.
Check if the
[compatible CUDA/CuDNN runtime](https://www.tensorflow.org/install/source#gpu) is installed.

## Crello experiments

Download and extract [Crello dataset](docs/crello-dataset.md). The following
script will download the dataset to `data/crello-dataset` directory.

```bash
bin/download_crello.sh
```

Prepare image data and learn a PixelVAE model for image embedding. The resulting
image encoder will be saved to `data/pixelvae/encoder`. This training takes
long. We recommend sufficient GPU resources to run this step (e.g., Tesla P100x4).

```bash
bin/generate_crello_image.sh
bin/train_pixelvae.sh
```

The training progress can be monitored via `tensorboard`:

```bash
tensorboard --logdir tmp/pixelvae/jobs
```

Once a PixelVAE is trained, build the crello document dataset, and learn
CanvasVAE models. The trainer script takes a few arguments to control
hyperparameters.
See `src/canvas-vae/canvasvae/main.py` for the list of available options.
This step can be run in a single GPU environment (e.g., Tesla P100x1).

```bash
bin/generate_crello_document.sh
bin/train_canvasvae.sh crello-document --latent-dim 512 --kl 32
```

The trainer outputs logs, evaluation results, and checkpoints to
`tmp/canvasvae/jobs/<job_id>`. The training progress can be monitored
via `tensorboard`:

```bash
tensorboard --logdir tmp/canvasvae/jobs
```

The resulting models can be further inspected in the notebook.

- `notebooks/crello-analysis.ipynb`

## RICO experiments

Download [UI SCREENSHOTS AND HIERARCHIES WITH SEMANTIC ANNOTATIONS](http://interactionmining.org/rico)
dataset first. This seems to require Google account. In the following, we assume
the downloaded archive file is placed in `tmp/rico_dataset_v0.1_semantic_annotations.zip`.

Once downloaded, preprocess and learn CanvasVAE models.

```bash
bin/generate_rico.sh tmp/rico_dataset_v0.1_semantic_annotations.zip
bin/train_canvasvae.sh rico --latent-dim 256 --kl 16
```

The resulting models can be inspected in the notebook.

- `notebooks/rico-analysis.ipynb`
