# User guide

The two algorithms in `pyaptamer` cover the two ends of an aptamer workflow:
scoring a known aptamer-protein pair, and generating new candidates for a target.

```{toctree}
:maxdepth: 1

aptanet
aptatrans
```

| Algorithm | Input | Output | API entry point |
| --- | --- | --- | --- |
| AptaNet | aptamer-protein sequence pairs | interaction label and probability | {class}`~pyaptamer.aptanet.AptaNetPipeline` |
| AptaTrans | target protein sequence | interaction score, candidate aptamers | {class}`~pyaptamer.aptatrans.AptaTransPipeline` |
