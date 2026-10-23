# WIKIBOOK
Combines the training datasets of wikitext2 and bookcorpus together.
- prepare.py takes arguments for the percentage of bookcorpus and wikitext2 to be used. it should be called from `nanoGPT_MQA\data\wikibook`
- prepare.py creates a train.bin and val.bin datasets. The metadata about the datasets are available in `dataset_info.txt`