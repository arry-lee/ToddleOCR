from unittest import TestCase


img = "../doc/imgs/11.jpg"
class TestModel(TestCase):

    def test_det(self):
        from ptocr.models.det.det_db_rvd import Model as DetModel
        from ptocr.models.rec.v3.rec_svtr_mv1e import Model as RecModel
        from ptocr.models.cls.v2.cls_cls_mv3 import Model as ClsModel
        dm = DetModel()
        # r = dm(img)
        rm = RecModel()
        cm = ClsModel()
        r = dm(img,rec=rm)
        print(r)

    def test_cls(self):
        from ptocr.models.cls.v2.cls_cls_mv3 import Model as ClsModel
        cm = ClsModel()
        r = cm("../doc/imgs_words/ch/word_2r.jpg")

    def test_det_v3(self):
        from ptocr.models.det.v3.det_db_mv3_rse import Model
        from ptocr.models.rec.v3.rec_svtr_mv1e import Model as RecModel
        m = Model(r'D:\dev\.model\toddleocr\zh_ocr_det_v3\inference.pt')
        rm = RecModel()
        r = m(img,rec=rm)
        print(r)

    def test_rec_v3(self):
        from ptocr.models.det.v3.det_db_mv3_rse import Model
        from ptocr.models.rec.v3.rec_svtr_mv1e import Model as RecModel
        m = Model(r'D:\dev\.model\toddleocr\zh_ocr_det_v3\inference.pt')
        rm = RecModel(r'D:\dev\.model\toddleocr\zh_ocr_rec_v3\inference.pt')
        r = m(img,rec=rm)
        print(r)
