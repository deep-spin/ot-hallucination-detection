# Optimal Transport for Unsupervised Hallucination Detection in Neural Machine Translation

This repository contains code and data for the "[Optimal Transport for Unsupervised Hallucination Detection in Neural Machine Translation](https://arxiv.org/abs/2212.09631)" paper accepted at ACL 2023 (main conference).

## Data with Hallucinations and NMT model

You can download the hallucinations corpus and the NMT model in the repo of the paper "[Looking for a Needle in a Haystack: A Comprehensive Study of Hallucinations in Neural Machine Translation](https://arxiv.org/abs/2208.05309)" paper accepted at EACL 2023 (main conference).

### Installation

First, start by installing the following dependencies:

```shell
python3 -m venv ot_env
source ot_env/bin/activate
pip install -r requirements.txt
```

## Replication of our experiments

First, start by downloading the data with statistics (attention weights, detection scores, and others):
```shell
https://web.tecnico.ulisboa.pt/~ist178550/ot-hallucination-detection-data.zip
```

After you unzip it, you will find two files (one with the stats for the hallucination corpus from Guerreiro et al. (2022) and one with stats for the heldout set used in our work -- also derived from the same work).

To replicate our experiments, just run the code on `run_detection.ipynb`.

If you want to replicate with our heldout data for Wass Combo, make sure to download the following file (Pandas dataframe):
```shell
https://web.tecnico.ulisboa.pt/~ist178550/data_heldout_for_thresholding.pkl
```

## If you found our work/code useful, please cite our work:
```bibtex
@misc{guerreiro2022optimal,
      title={Optimal Transport for Unsupervised Hallucination Detection in Neural Machine Translation}, 
      author={Nuno M. Guerreiro and Pierre Colombo and Pablo Piantanida and André F. T. Martins},
      year={2022},
      eprint={2212.09631},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
