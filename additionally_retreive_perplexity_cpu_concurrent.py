from config import init_polish_perplexity_model, init_nltk

from concurrent.futures import ProcessPoolExecutor, as_completed

from typing import List
from tqdm import tqdm

from dao.attribute import DAOAttributePL

from models.attribute import AttributePLInDB

from analysis.attribute_retriving import calculate_perplexity
from services.utils import suppress_stdout

def calculate_perplexity_concurrent(attribute_in_db: AttributePLInDB):
    text_to_analyse = attribute_in_db.stylometrix_metrics.text
    with suppress_stdout():
        perplexity_base, perplexity = calculate_perplexity(text_to_analyse, 'pl', return_both=True, force_use_cpu=True)
    return attribute_in_db.id, perplexity, perplexity_base


def process(attributes_db: str):
    dao_attributes = DAOAttributePL(attributes_db)

    attributes_to_process: List[AttributePLInDB] = dao_attributes.find_many_by_query({'perplexity': None})
    with ProcessPoolExecutor() as executor:
        # Step 1: Preprocess real lab reports
        tasks = [executor.submit(calculate_perplexity_concurrent, attribute_in_db) for attribute_in_db in attributes_to_process]

        for future in tqdm(as_completed(tasks), desc="Calculating perplexity", total=len(tasks)):
            try:
                attribute_data_to_update = future.result()
                dao_attributes.update_one({'_id': attribute_data_to_update[0]},
                                          {"$set": {'perplexity_base': attribute_data_to_update[2],
                                                          'perplexity': attribute_data_to_update[1]}})
            except Exception as e:
                print(e)


if __name__ == "__main__":
    init_nltk()
    init_polish_perplexity_model()
    attributes_db_name = ''

    process(attributes_db=attributes_db_name)