
from model.caser import CaserModel
from model.gru4rec import GRU4RecModel
from model.sasrec import SASRecModel
from model.bert4rec import BERT4RecModel
from model.sfsrec import SFSRecModel 


MODEL_DICT = {
    "caser": CaserModel,
    "gru4rec": GRU4RecModel,
    "sasrec": SASRecModel,
    "bert4rec": BERT4RecModel,
    "sfsrec": SFSRecModel,  

    }