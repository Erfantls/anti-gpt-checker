from analysis.nlp_transformations import preprocess_text
from config import init_language_tool_pl, init_language_tool_en

from dao.lab_report import DAOLabReport

from models.lab_report import LabReportInDB

from analysis.attribute_retriving import spelling_and_grammar_check

if __name__ == "__main__":
    init_language_tool_pl()
    init_language_tool_en()
    dao_lab_reports = DAOLabReport(collection_name="")
    report_to_analyse: LabReportInDB = dao_lab_reports.find_by_id("67546cd3f8817ec8fd3332b6") #67546cd4f8817ec8fd3332b7 67546cd3f8817ec8fd3332b6
    text_to_analyse = preprocess_text(report_to_analyse.plaintext_content)
    text_errors_by_category, number_of_errors, number_of_abbreviations, number_of_unrecognized_words = spelling_and_grammar_check(text_to_analyse, 'pl')
    print(text_errors_by_category, number_of_errors)