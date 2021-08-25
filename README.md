[Dataset](docs/crello-dataset.md) | [arXiv](https://arxiv.org/abs/2108.01249)

# CanvasVAE

Official tensorflow implementation of the following work.

> Kota Yamaguchi, CanvasVAE: Learning to Generate Vector Graphic Documents, ICCV 2021

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

## Crello experiments

Download and extract [Crello dataset](docs/crello-dataset.md). The following
script will download the dataset to `data/crello-dataset` directory.

```bash
bin/download_crello.sh
```

Prepare image data and learn a PixelVAE model for image embedding. The resulting
image encoder will be saved to `data/pixelvae/encoder`.

```bash
bin/generate_crello_image.sh
bin/train_pixelvae.sh
```

Build a crello document dataset, and learn CanvasVAE models. The trainer script
takes a few arguments to control hyperparameters. See
`src/canvas-vae/canvasvae/main.py` for the list of available options.

```bash
bin/generate_crello_document.sh
bin/train_canvasvae.sh crello-document --latent-dim 512 --kl 16
```

The trainer outputs logs, evaluation results, and checkpoints to
`tmp/canvasvae/jobs/<job_id>`. The resulting models can be further inspected in
the notebook.

- `notebooks/crello-analysis.ipynb`

## RICO experiments

Prepare the [RICO dataset](https://interactionmining.org/rico), and learn
CanvasVAE models.

```bash
bin/generate_rico.sh
bin/train_canvasvae.sh rico --latent-dim 256 --kl 16
```

The resulting models can be inspected in the notebook.

- `notebooks/rico-analysis.ipynb`
