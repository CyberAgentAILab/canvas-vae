# Preprocess

Dataflow jobs to generate TFRecord datasets.

## Crello

Job to generate Crello dataset from scraped results from `crello-scraping/`.

```bash
./bin/generate_crello.sh
```

## Crello Image

Job to generate Crello image dataset for training PixelVAE model.

```bash
./bin/generate_crello_image.sh
```

## Crello Document

Job to generate Crello document dataset. This job requires Crello dataset and
a trained PixelVAE model.

```bash
./bin/generate_crello_document.sh
```

## RICO

Job to generate RICO dataset from the zip archive.

```bash
./bin/generate_rico.sh
```

## Magazine

Experimental job to generate Magazine dataset. Download the dataset archive
before running the job.

```bash
./bin/generate_rico.sh
```
