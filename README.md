# ROP 299

This repo includes all the files created and used for this ROP. 
Files can be used by following below steps.

1. using Generate_txt.py, we preprocess survey.csv by writing processed contents we need into a txt file: survey.txt.
2. running demo.sh file in GloVe-Master folder will generate results of GloVe embedding from survey.txt file. 
3. step2.py includes functions that generates TSNE visualization of obtained GloVe embedding, and topic modelling algorithm LDA.

GloVe embedding results and LDA topic modelling results are also saved in folders for view. 
