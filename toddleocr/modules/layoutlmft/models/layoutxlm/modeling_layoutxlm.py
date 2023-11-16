# coding=utf-8
from transformers.utils import logging

from ..layoutlmv2 import LayoutLMv2ForRelationExtraction, LayoutLMv2ForTokenClassification, LayoutLMv2Model
from .configuration_layoutxlm import LayoutXLMConfig


logger = logging.get_logger(__name__)

LAYOUTXLM_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "layoutxlm-base",
    "layoutxlm-large",
]


class LayoutXLMModel(LayoutLMv2Model):
    config_class = LayoutXLMConfig
    pretrained_model_archive_map = LAYOUTXLM_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "layoutxlm"

class LayoutXLMForTokenClassification(LayoutLMv2ForTokenClassification):
    config_class = LayoutXLMConfig
    pretrained_model_archive_map = LAYOUTXLM_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "layoutxlm"
    base_model = LayoutXLMModel


class LayoutXLMForRelationExtraction(LayoutLMv2ForRelationExtraction):
    config_class = LayoutXLMConfig
    pretrained_model_archive_map = LAYOUTXLM_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "layoutxlm"
    base_model = LayoutXLMModel
