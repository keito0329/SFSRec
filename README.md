# SFSRec
This is the official source code for our SFSRec submitted to ECIR2026.


## Dataset
In our experiments, we utilize five datasets, all stored in the `src/data` folder. 
- For the Sports, Toys, Beauty, Yelp, LastFM datasets, we employed the datasets downloaded from [this repository](https://github.com/Woeee/FMLP-Rec). 


### How to train SFSRec
- Note that pretrained model (.pt) and train log file (.log) will saved in `src/output`
- `train_name`: name for log file and checkpoint file
```
python main.py  --data_name [DATASET] \
                --lr [LEARNING_RATE] \
                --train_name [LOG_NAME]
```
- Example for Beauty
```
python main.py  --data_name Beauty \
                --lr 0.0005 \
                --train_name SFSRec_Beauty
```

## Acknowledgement
This repository is based on [BSARec](https://github.com/yehjin-shin/BSARec).
