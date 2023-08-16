# basic loss function
from .basic_loss import DistanceLoss

# basic_loss
from .basic_loss import LossFromOutput

# cls loss
from .cls_loss import ClsLoss

# combined loss function
from .combined_loss import CombinedLoss
from .det_ct_loss import CTLoss

# det loss
from .det_db_loss import DBLoss
from .det_drrg_loss import DRRGLoss
from .det_east_loss import EASTLoss
from .det_fce_loss import FCELoss
from .det_pse_loss import PSELoss
from .det_sast_loss import SASTLoss

# e2e loss
from .e2e_pg_loss import PGLoss
from .kie_sdmgr_loss import SDMGRLoss
from .rec_aster_loss import AsterLoss
from .rec_att_loss import AttentionLoss
from .rec_can_loss import CANLoss
from .rec_ce_loss import CELoss

# rec loss
from .rec_ctc_loss import CTCLoss
from .rec_multi_loss import MultiLoss
from .rec_pren_loss import PRENLoss
from .rec_rfl_loss import RFLLoss
from .rec_sar_loss import SARLoss
from .rec_spin_att_loss import SPINAttentionLoss
from .rec_srn_loss import SRNLoss
from .rec_vl_loss import VLLoss

# sr loss
from .stroke_focus_loss import StrokeFocusLoss

# table loss
from .table_att_loss import SLALoss, TableAttentionLoss
from .table_master_loss import TableMasterLoss
from .text_focus_loss import TelescopeLoss

# vqa token loss
from .vqa_token_layoutlm_loss import VQASerTokenLayoutLMLoss

from torchvision.ops import box_iou

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
    "MultiLoss",
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
