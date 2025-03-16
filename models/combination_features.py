from typing import Optional

from pydantic import BaseModel

from models.stylometrix_metrics import AllStyloMetrixFeaturesPL, AllStyloMetrixFeaturesEN


class CombinationFeatures(BaseModel):
    content_function_ratio: Optional[float]
    common_long_word_ratio: Optional[float]
    common_rare_word_ratio: Optional[float]
    active_passive_voice_ratio: Optional[float]

    @staticmethod
    def init_from_stylometrix(stylometrix_metrics: AllStyloMetrixFeaturesPL | AllStyloMetrixFeaturesEN):
        content_function_ratio = stylometrix_metrics.lexical.L_CONT_A / (
            stylometrix_metrics.lexical.L_FUNC_A if stylometrix_metrics.lexical.L_FUNC_A else 1)
        common_long_word_ratio = stylometrix_metrics.lexical.L_TCCT1 / (
            stylometrix_metrics.lexical.L_SYL_G4 if stylometrix_metrics.lexical.L_SYL_G4 else 1)
        common_rare_word_ratio = stylometrix_metrics.lexical.L_TCCT1 / (
            1 - stylometrix_metrics.lexical.L_TCCT5 if stylometrix_metrics.lexical.L_SYL_G4 else 1)
        active_passive_voice_ratio = stylometrix_metrics.inflection.IN_V_ACT / (
            stylometrix_metrics.inflection.IN_V_PASS if stylometrix_metrics.inflection.IN_V_PASS else 1)
        return CombinationFeatures(
            content_function_ratio=content_function_ratio,
            common_long_word_ratio=common_long_word_ratio,
            common_rare_word_ratio=common_rare_word_ratio,
            active_passive_voice_ratio=active_passive_voice_ratio
        )
