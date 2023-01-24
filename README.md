# NEWS-COPY

Code for our paper "Noise-Robust De-Duplication at Scale"
[[NBER](https://www.nber.org/papers/w30726)], [[arxiv](https://arxiv.org/abs/2210.04261)], [[ICLR](https://openreview.net/forum?id=bAz2DBS35i)]

This repo includes: 
- NEWS-COPY dataset: 7,210 document dataset, with 122,876 positive duplicate pairs, for studying noise-robust de-duplication. 
- Rule-based de-duplication methods: hashing and N-gram overlap. 
- Neural de-duplication methods: a contrastively trained bi-encoder, and a "re-rank" style approach combining a bi- and cross-encoder. 
- Inference at scale for hashing and the biencoder methods.  

If you find this work useful, please cite the following paper: 

    @inproceedings{silcock-etal-2020-noise,
      title = "Noise-Robust De-Duplication at Scale",
      author = "Silcock, Emily and D'Amico-Wong, Luca and Yang, Jinglin and Dell, Melissa",
      booktitle = "International Conference on Learning Representations (ICLR)",
      year = "2023",
    }

### Installation

    git clone https://github.com/dell-research-harvard/NEWS-COPY.git
    cd NEWS-COPY
    conda env create -f environment.yml


### Data
- Historical Newspapers: train, evaluation and test sets can be downloaded [here](https://www.dropbox.com/sh/so3iw4xecayyrow/AAAiy5FhDf0WpUeHFzxO1SIza?dl=0). For more detail see the paper above 
- C4: C4 can be downloaded thanks to AllenAI - see https://github.com/allenai/allennlp/discussions/5056


### Rule-based
Codebase for neural methods for de-duplication, n-gram overlap and locally sensitive hashing. These predominate in the literature, but are significantly outperformed by the neural methods below. 


### Neural
Training and evaluation scripts for a contrastively trained bi-encoder, and a "re-rank" style approach combining a bi- and cross-encoder. These outperform the rule-based approaches above. 


### Inference at scale
Inference at scale for hashing (LSH) and the biencoder methods over C4 and SuperGlue. The bi-encoder scales well, de-duplicating a 10 million text corpus on a single GPU card in a matter of hours. 
