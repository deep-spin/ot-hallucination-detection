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

After you unzip it, you will find two files (one with the stats for the hallucination corpus from [Guerreiro et al. (2023)](https://arxiv.org/abs/2208.05309) and one with stats for the heldout set used in our work -- also derived from the same work).

To replicate our experiments, just run the code on `run_detection.ipynb`.

If you want to replicate our results with the heldout data for Wass Combo, make sure to download the following file (Pandas dataframe):
```shell
https://web.tecnico.ulisboa.pt/~ist178550/data_heldout_for_thresholding.pkl
```

## If you found our work/code useful, please consider citing our work:
```bibtex
@inproceedings{guerreiro-etal-2023-optimal,
    title = "Optimal Transport for Unsupervised Hallucination Detection in Neural Machine Translation",
    author = "Guerreiro, Nuno M.  and
      Colombo, Pierre  and
      Piantanida, Pablo and
      Martins, Andr{\'e}",
    booktitle = "Proceedings of The 61st Annual Meeting of the Association for Computational Linguistics",
    month = "july",
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
}
```
```bibtex
@inproceedings{guerreiro-etal-2023-looking,
    title = "Looking for a Needle in a Haystack: A Comprehensive Study of Hallucinations in Neural Machine Translation",
    author = "Guerreiro, Nuno M.  and
      Voita, Elena  and
      Martins, Andr{\'e}",
    booktitle = "Proceedings of the 17th Conference of the European Chapter of the Association for Computational Linguistics",
    month = may,
    year = "2023",
    address = "Dubrovnik, Croatia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.eacl-main.75",
}
```
