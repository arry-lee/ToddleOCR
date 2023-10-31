# basic loss function
from .aster import AsterLoss
from .att import AttentionLoss

# basic_loss
from .basic import DistanceLoss, LossFromOutput
from .can import CANLoss
from .ce import CELoss

# cls loss
from .cls import ClsLoss

# combined loss function
from .compose import CombinedLoss, MultiLoss
from .ct import CTLoss

# rec loss
from .ctc import CTCLoss

# det loss
from .db import DBLoss
from .drrg import DRRGLoss
from .east import EASTLoss
from .fce import FCELoss

# e2e loss
from .pg import PGLoss
from .pren import PRENLoss
from .pse import PSELoss
from .rfl import RFLLoss
from .sar import SARLoss
from .sast import SASTLoss
from .sdmgr import SDMGRLoss
from .spin import SPINAttentionLoss
from .srn import SRNLoss

# sr loss
from .stroke_focus import StrokeFocusLoss

# table loss
from .table_att import SLALoss, TableAttentionLoss
from .table_master import TableMasterLoss
from .text_focus import TelescopeLoss
from .vl import VLLoss

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
