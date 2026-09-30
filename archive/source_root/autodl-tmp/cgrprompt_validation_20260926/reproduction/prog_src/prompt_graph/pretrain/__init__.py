from .Bandana import Bandana
from .DGI import DGI
from .Edgepred_GPPT import Edgepred_GPPT
from .Edgepred_Gprompt import Edgepred_Gprompt
from .GraphCL import GraphCL
from .GraphMAE import GraphMAE
from .GraphMAE2 import GraphMAE2
from .MultiGPrompt import GraphPrePrompt, NodePrePrompt, prompt_pretrain_sample
from .S2GAE import S2GAE
from .SimGRACE import SimGRACE

__all__ = [
    "DGI",
    "Bandana",
    "Edgepred_GPPT",
    "Edgepred_Gprompt",
    "GraphCL",
    "GraphMAE",
    "GraphMAE2",
    "GraphPrePrompt",
    "NodePrePrompt",
    "SimGRACE",
    "S2GAE",
    "prompt_pretrain_sample",
]
