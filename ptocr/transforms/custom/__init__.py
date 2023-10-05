from .rec_img_aug import (
    BaseDataAugmentation,
    RecAug,
    RecConAug,
    RecResizeImg,
    ClsResizeImg,
    SRNRecResizeImg,
    GrayRecResizeImg,
    SARRecResizeImg,
    PRENResizeImg,
    ABINetRecResizeImg,
    SVTRRecResizeImg,
    ABINetRecAug,
    VLRecResizeImg,
    SPINRecResizeImg,
    RobustScannerRecResizeImg,
    RFLRecResizeImg,
    SVTRRecAug,
)

from .make_pse_gt import MakePseGt



from .operators import *
from .label_ops import *

from .east_process import *
from .sast_process import *
from .pg_process import *
from .table_ops import *

from .vqa import *

from .fce_aug import *
from .fce_targets import FCENetTargets
