# basic loss function
from .basic import DistanceLoss

# basic_loss
from .basic import LossFromOutput

# cls loss
from .cls import ClsLoss

# combined loss function
from .compose import CombinedLoss, MultiLoss
from .ct import CTLoss

# det loss
from .db import DBLoss
from .drrg import DRRGLoss
from .east import EASTLoss
from .fce import FCELoss
from .pse import PSELoss
from .sast import SASTLoss

# e2e loss
from .pg import PGLoss
from .sdmgr import SDMGRLoss
from .aster import AsterLoss
from .att import AttentionLoss
from .can import CANLoss
from .ce import CELoss

# rec loss
from .ctc import CTCLoss
from .pren import PRENLoss
from .rfl import RFLLoss
from .sar import SARLoss
from .spin import SPINAttentionLoss
from .srn import SRNLoss
from .vl import VLLoss

# sr loss
from .stroke_focus import StrokeFocusLoss

# table loss
from .table_att import SLALoss, TableAttentionLoss
from .table_master import TableMasterLoss
from .text_focus import TelescopeLoss

# vqa token loss
from .vqa_token_layoutlm import VQASerTokenLayoutLMLoss

__all__ = [
    "DBLoss",
    "PSELoss",
    "EASTLoss",
    "SASTLoss",
    "FCELoss",
    "CTCLoss",
    "ClsLoss",
    "AttentionLoss",
    "SRNLoss",
    "PGLoss",
    "CombinedLoss",
    "CELoss",
    "TableAttentionLoss",
    "SARLoss",
    "AsterLoss",
    "SDMGRLoss",
    "VQASerTokenLayoutLMLoss",
    "LossFromOutput",
    "PRENLoss",
    "TableMasterLoss",
    "SPINAttentionLoss",
    "VLLoss",
    "StrokeFocusLoss",
    "SLALoss",
    "CTLoss",
    "RFLLoss",
    "DRRGLoss",
    "CANLoss",
    "TelescopeLoss",
]
