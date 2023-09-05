import torch

m = torch.load(r'D:\dev\.model\toddleocr\te_ocr_rec_v3\inference0.pt')

torch.save(m,r'D:\dev\.model\toddleocr\te_ocr_rec_v3\inference.pt',pickle_protocol=-1)
