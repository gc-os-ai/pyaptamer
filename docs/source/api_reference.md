# API reference

This page documents the public API. Names not listed here are internal and may
change without notice.

## Data

```{eval-rst}
.. currentmodule:: pyaptamer.data

.. autosummary::
   :toctree: generated
   :nosignatures:

   MoleculeLoader
```

## Transformations

```{eval-rst}
.. currentmodule:: pyaptamer.transformations.transform

.. autosummary::
   :toctree: generated
   :nosignatures:

   DNAToRNA
   PrimerTrimmer
```

## Compose

```{eval-rst}
.. currentmodule:: pyaptamer.transformations.compose

.. autosummary::
   :toctree: generated
   :nosignatures:

   ApplyToCols
```

## DeepDNAShape

```{eval-rst}
.. currentmodule:: pyaptamer.deepdnashape

.. autosummary::
   :toctree: generated
   :nosignatures:

   DeepDNAShape
```

## AptaNet

```{eval-rst}
.. currentmodule:: pyaptamer.aptanet

.. autosummary::
   :toctree: generated
   :nosignatures:

   AptaNetPipeline
   AptaNetClassifier
   AptaNetRegressor
```

## AptaTrans

```{eval-rst}
.. currentmodule:: pyaptamer.aptatrans

.. autosummary::
   :toctree: generated
   :nosignatures:

   AptaTransPipeline
   AptaTrans
   AptaTransLightning
   AptaTransEncoderLightning
   EncoderPredictorConfig
```

## DeepAptamer

```{eval-rst}
.. currentmodule:: pyaptamer.deepaptamer
   DeepAptamerPipeline
   DeepAptamerClassifier
   DeepAptamerFeatures
   DeepAptamerNN
```

## AptaMCTS

```{eval-rst}
.. currentmodule:: pyaptamer.aptamcts

.. autosummary::
   :toctree: generated
   :nosignatures:

   AptaMCTSPipeline

## AptaDiff

```{eval-rst}
.. currentmodule:: pyaptamer.aptadiff

.. autosummary::
   :toctree: generated
   :nosignatures:

   AptaDiffGenerator
   AptaDiffLightning
   AptaDiffDiffusion
   AptaDiffDenoiser
```

## RaptGen

```{eval-rst}
.. currentmodule:: pyaptamer.raptgen

.. autosummary::
   :toctree: generated
   :nosignatures:

   RaptGenGenerator
   RaptGenModel
   RaptGenLightning
   CNNPHMMVAE
   CNNPHMMVAEFast
```

```

## Monte Carlo Tree Search

```{eval-rst}
.. currentmodule:: pyaptamer.mcts

.. autosummary::
   :toctree: generated
   :nosignatures:

   MCTS
```

## Experiments

Scoring functions that connect a model to the search.

```{eval-rst}
.. currentmodule:: pyaptamer.experiments

.. autosummary::
   :toctree: generated
   :nosignatures:

   AptamerEvalAptaMCTS
   AptamerEvalAptaNet
   AptamerEvalAptaTrans
```

## Encodings

```{eval-rst}
.. currentmodule:: pyaptamer.transformations.encode

.. autosummary::
   :toctree: generated
   :nosignatures:

   ConjointTriad
   GreedyEncoder
   KMerFrequencies
   PSeAAC
   SequenceIndexEncoder
   SequenceOneHotEncoder
```

## Datasets

```{eval-rst}
.. currentmodule:: pyaptamer.datasets

.. autosummary::
   :toctree: generated
   :nosignatures:

   load_aptacom_full
   load_aptacom
   load_csv_dataset
   load_from_rcsb
   load_hf_to_dataset
   load_1brq
   load_1gnh
   load_5nu7
   load_li2014
   load_sample_fastq
```

## Benchmarking

```{eval-rst}
.. currentmodule:: pyaptamer.benchmarking

.. autosummary::
   :toctree: generated
   :nosignatures:

   Benchmarking
```

## Utilities

```{eval-rst}
.. currentmodule:: pyaptamer.utils

.. autosummary::
   :toctree: generated
   :nosignatures:

   aa_str_to_letter
   dna2rna
   encode_rna
   generate_nplets
   rna2vec
   pdb_to_seq_uniprot
```
