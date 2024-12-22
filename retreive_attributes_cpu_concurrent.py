from config import init_polish_perplexity_model, init_spacy_polish_nlp_model, init_language_tool_pl, \
    init_language_tool_en, init_nltk

from concurrent.futures import ProcessPoolExecutor, as_completed

from typing import List
from tqdm import tqdm

from dao.lab_report import DAOLabReport
from dao.attribute import DAOAttributePL

from models.lab_report import LabReportInDB
from models.attribute import AttributePL

from analysis.attribute_retriving import perform_full_analysis
from analysis.nlp_transformations import preprocess_text
from services.utils import suppress_stdout

def process_file(report: LabReportInDB, is_generated: bool):
    text_to_analyse = preprocess_text(report.plaintext_content)
    with suppress_stdout():
        analysis_result = perform_full_analysis(
            text=text_to_analyse,
            lang_code='pl',
            skip_perplexity_calc=True,
            skip_stylometrix_calc=True
        )

    attribute_to_insert = AttributePL(
        referenced_db_name='lab_reports',
        referenced_doc_id=report.id,
        language="pl",
        is_generated=is_generated,
        is_personal=None,
        **analysis_result.dict()
    )

    return attribute_to_insert


def process(report_db: str, attributes_db: str):
    dao_lab_reports = DAOLabReport(report_db)
    dao_attributes = DAOAttributePL(attributes_db)

    real_lab_reports: List[LabReportInDB] = dao_lab_reports.find_many_by_query({'is_generated': False})
    gen_lab_reports: List[LabReportInDB] = dao_lab_reports.find_many_by_query({'is_generated': True})
    alreadyprocessed_lab_reports = dao_attributes.find_many_by_query({})
    alreadyprocessed_lab_reports_ids = [report.referenced_doc_id for report in alreadyprocessed_lab_reports]
    real_lab_reports = [report for report in real_lab_reports if report.id not in alreadyprocessed_lab_reports_ids]
    gen_lab_reports = [report for report in gen_lab_reports if report.id not in alreadyprocessed_lab_reports_ids]

    with ProcessPoolExecutor() as executor:
        # Step 1: Preprocess real lab reports
        tasks = [executor.submit(process_file, lab_report, False) for lab_report in real_lab_reports]

        for future in tqdm(as_completed(tasks), desc="Real lab", total=len(tasks)):
            try:
                attribute_to_insert = future.result()
                dao_attributes.insert_one(attribute_to_insert)
            except Exception as e:
                print(e)

        tasks = [executor.submit(process_file, lab_report, True) for lab_report in gen_lab_reports]
        for future in tqdm(as_completed(tasks), desc="Gen lab", total=len(tasks)):
            try:
                attribute_to_insert = future.result()
                dao_attributes.insert_one(attribute_to_insert)
            except Exception as e:
                print(e)


if __name__ == "__main__":
    init_nltk()
    init_spacy_polish_nlp_model()
    #init_polish_perplexity_model()
    init_language_tool_pl()
    init_language_tool_en()
    report_db_name = 'lab_reports-24-12-16'
    attributes_db_name = 'attributes-24-12-16-recalc-24-12-21.N'

    process(report_db=report_db_name, attributes_db=attributes_db_name)