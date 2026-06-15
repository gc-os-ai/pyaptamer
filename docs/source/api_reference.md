# API reference

This page documents the public API. Names not listed here are internal and may
change without notice.

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

   AptamerEvalAptaNet
   AptamerEvalAptaTrans
```

## Encodings

```{eval-rst}
.. currentmodule:: pyaptamer.pseaac

.. autosummary::
   :toctree: generated
   :nosignatures:

   PSeAAC
   AptaNetPSeAAC
```

## Datasets

```{eval-rst}
.. currentmodule:: pyaptamer.datasets

.. autosummary::
   :toctree: generated
   :nosignatures:

   load_aptacom_full
   load_aptacom_x_y
   load_csv_dataset
   load_from_rcsb
   load_hf_to_dataset
   load_1brq
   load_1gnh
   load_5nu7
   load_li2014
   load_pfoa
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
   pdb_to_struct
   struct_to_aaseq
   pdb_to_seq_uniprot
   pdb_to_aaseq
```
