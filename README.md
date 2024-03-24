# Hit Discovery for novel antimalarial Using AI

This project seeks to use AI to discover novel drugs for anti-malarial.

Steps/Process
[] Get all anti-malaria dataset from CHEMBL
[] Use LoHi splitter to train model and identifiy antimalaria
[] Try multiple models to predict whether antimalaria (gradient boosted,random forest, ANN, logistic regression, GCNN, GAT, Chemprop, MolChem)
[] Train CLM to predict new molecules (LSTM, GRU, RNN, Transformers, Mamba, VAE) on a large CHEMBL dataset
[] Finetune on subselected positive malaria datasets. 
[] Test the molecule to identify which are positive.
[] Dock positive molecules.
[] Synthesis and perform wetlab evaluation of the positive.