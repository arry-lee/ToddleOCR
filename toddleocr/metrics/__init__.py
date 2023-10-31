from .cls import ClsMetric
from .ct import CTMetric
from .det import DetFCEMetric, DetMetric
from .distillation import DistillationMetric
from .e2e import E2EMetric
from .kie import KIEMetric
from .rec import CANMetric, CNTMetric, RecMetric
from .sr import SRMetric
from .table import TableMetric
from .vqa import VQAReTokenMetric, VQASerTokenMetric

__all__ = [
    "DetMetric",
    "DetFCEMetric",
    "RecMetric",
    "ClsMetric",
    "E2EMetric",
    "DistillationMetric",
    "TableMetric",
    "KIEMetric",
    "VQAReTokenMetric",
    "SRMetric",
    "CTMetric",
    "CNTMetric",
    "CANMetric",
]
