from typing import List

import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import pyplot as plt

from api.api_models.response import HistogramData, HistogramDataDTO
from api.server_config import API_ATTRIBUTES_REFERENCE_COLLECTION_NAME, API_MONGODB_DB_NAME, API_HISTOGRAMS_PATH
from api.api_models.lightbulb_score import LightbulbScoreType
from dao.attribute import DAOAttributePL
from models.attribute import AttributePLInDB

dao_attribute_reference: DAOAttributePL = DAOAttributePL(collection_name=API_ATTRIBUTES_REFERENCE_COLLECTION_NAME, db_name=API_MONGODB_DB_NAME)

GENERATED_FLAT_DICT = None
REAL_FLAT_DICT = None

def load_reference_attributes() -> None:
    global GENERATED_FLAT_DICT, REAL_FLAT_DICT
    if GENERATED_FLAT_DICT is not None and REAL_FLAT_DICT is not None:
        return
    generated: List[AttributePLInDB] = dao_attribute_reference.find_many_by_query({"is_generated": True})
    real: List[AttributePLInDB] = dao_attribute_reference.find_many_by_query({"is_generated": False})

    GENERATED_FLAT_DICT = [(x.to_flat_dict_normalized(), 1) for x in generated]
    REAL_FLAT_DICT = [(x.to_flat_dict_normalized(), 0) for x in real]
    print(f"LOADED {len(GENERATED_FLAT_DICT) + len(REAL_FLAT_DICT)} attributes from reference collection")

def plot_two_hists(data1, data2, title, metric_name="Metric", num_bin=21, min_value=0, max_value=5, top=0.5,
                   additional_value=None, file_name=""):
    # Truncate data to max_value if needed
    data1_to_plot = [d if d < max_value else max_value for d in data1]
    data2_to_plot = [d if d < max_value else max_value for d in data2]

    w = (max_value - min_value) / num_bin
    bins = np.arange(min_value, max_value + w, w)

    weights1 = np.ones_like(data1_to_plot) / len(data1_to_plot)
    weights2 = np.ones_like(data2_to_plot) / len(data2_to_plot)

    plt.hist(data1_to_plot, bins=bins, weights=weights1, alpha=0.7, label='Generated', color='red')
    plt.hist(data2_to_plot, bins=bins, weights=weights2, alpha=0.7, label='Real', color='blue')
    if additional_value is not None:
        plt.axvline(additional_value, color='red', linestyle='--', linewidth=1, label='Sample value')

    plt.title(title)
    plt.xlim([min_value, max_value])
    plt.ylim(top=top)
    plt.xlabel(f'{metric_name} value')
    plt.ylabel('Lab reports share')
    plt.legend()
    plt.savefig(f'{API_HISTOGRAMS_PATH}/{file_name}.png')
    plt.clf()

def compute_histogram_data(attribute_name: str, num_bin=21,
                           min_value=None, max_value=None, additional_value=None) -> HistogramDataDTO:
    data_gen = [attribute[0][attribute_name] for attribute in GENERATED_FLAT_DICT]
    data_real = [attribute[0][attribute_name] for attribute in REAL_FLAT_DICT]

    if min_value is None:
        min_value = 0
    if max_value is None:
        max_value = max(np.percentile(data_gen, 95), np.percentile(data_real, 95))

    # Clip values
    data_gen = np.clip(data_gen, None, max_value)
    data_real = np.clip(data_real, None, max_value)

    w = (max_value - min_value) / num_bin
    bins = np.arange(min_value, max_value + w, w).tolist()

    counts_gen, _ = np.histogram(data_gen, bins=bins)
    counts_real, _ = np.histogram(data_real, bins=bins)

    histogram_llm = HistogramData(
        feature=attribute_name,
        data_type="llm-generated",
        bins=bins,
        counts=counts_gen.tolist()
    )

    histogram_human = HistogramData(
        feature=attribute_name,
        data_type="human-written",
        bins=bins,
        counts=counts_real.tolist()
    )

    dto = HistogramDataDTO(
        llm=histogram_llm,
        human=histogram_human,
        additional_value=additional_value,
        min_value=min_value,
        max_value=max_value,
        num_bins=num_bin,
        object_hash = ""
    )

    dto.object_hash = dto.calculate_histogram_hash()

    return dto



def compare_2_hists(attribute_name: str, min_value=None, max_value=None, top=0.41, num_bin=21,
                    additional_value=None, file_name:str= "", title:str= "") -> None:
    data_gen = [attribute[0][attribute_name] for attribute in GENERATED_FLAT_DICT]
    data_real = [attribute[0][attribute_name] for attribute in REAL_FLAT_DICT]
    if min_value is None:
        min_value = 0  # min(min(data_gen), min(data_real))
    if max_value is None:
        max_value = max(np.percentile(data_gen, 95), np.percentile(data_real, 95))

    plot_two_hists(data_gen, data_real, title=title, metric_name=attribute_name,
                   min_value=min_value, max_value=max_value, top=top, num_bin=num_bin,
                   additional_value=additional_value, file_name=file_name)


def _relative_density(value: float,
                      real_kde: gaussian_kde,
                      gen_kde:  gaussian_kde) -> float:
    """
    Raw score in [-1,1]:  +1 → the value sits only under the human curve,
                          -1 → the value sits only under the LLM curve,
                           0 → equally plausible under both.
    """
    p_real = real_kde.evaluate([value])[0]
    p_gen  = gen_kde.evaluate([value])[0]

    if p_real + p_gen == 0:        # totally unseen value
        return 0.0

    return (p_real - p_gen) / (p_real + p_gen)   # ∈ (-1,1)


def calculate_lightbulb_score(attribute_value,
                              attribute_name,
                              category=LightbulbScoreType.BIDIRECTIONAL,
                              bandwidth='scott') -> float:
    """
    Returns a scalar whose range depends on *category*.

    BIDIRECTIONAL : [-1, 1]   (+ → human-like, − → LLM-like)
    HUMAN_WRITTEN : [-1, 0]   (close to -1 → confidently human)
    LLM_GENERATED : [ 0, 1]   (close to  1 → confidently LLM)
    """
    gen_values = [attribute[0][attribute_name] for attribute in GENERATED_FLAT_DICT]
    real_values = [attribute[0][attribute_name] for attribute in REAL_FLAT_DICT]

    # KDEs give smooth non-parametric estimates; 1-liner to swap in any other model.
    real_kde = gaussian_kde(real_values, bw_method=bandwidth)
    gen_kde  = gaussian_kde(gen_values,  bw_method=bandwidth)

    raw = _relative_density(attribute_value, real_kde, gen_kde)  # [-1,1]

    if category == LightbulbScoreType.BIDIRECTIONAL:
        return float(np.clip(raw, -1, 1))

    if category == LightbulbScoreType.HUMAN_WRITTEN:
        human_score = -raw              # make human side negative
        return float(np.clip(human_score, -1, 0))

    if category == LightbulbScoreType.LLM_GENERATED:
        llm_score =  raw                # keep LLM side positive
        return float(np.clip(llm_score,  0, 1))

    raise ValueError(f"Unknown category: {category}")