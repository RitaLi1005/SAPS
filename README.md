# SAPS
Code Replication for "An Empirical Study of Sample Selection Strategies for Large Language Model Repair"

This repository contains code related to the paper _An Empirical Study of Sample Selection Strategies for Large Language Model Repair_. The framework is shown as follows:
![image](image/frameworkrq.pdf)


## Process of SAPS

(a) Embedding Extraction, where input samples are mapped into semantic embedding vectors using a fixed pre-trained encoder, ensuring model-agnostic prioritization; 

(b) Representation Structuring, where embeddings are reduced in dimensionality and clustered to uncover coarse-grained semantic structures;

(c) Boundary-Aware Sampling, which selects peripheral samples within each cluster to capture semantic boundaries and preserve informative diversity. The resulting prioritized dataset balances efficiency and representativeness for downstream model repair.

The framework is shown as follows:
![image](image/framework.pdf)


## Acknowledgement
