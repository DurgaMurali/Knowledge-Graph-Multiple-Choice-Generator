
import argparse
import os

def remove_multiple_answers(qa_dataset_path, qa_dataset_processed_path):
    print(qa_dataset_path, ", ", qa_dataset_processed_path)
    num_lines = 0
    num_lines_with_ans = 0
    num_lines_with_one_ans = 0
    with open(qa_dataset_path, 'rt', encoding='utf-8') as inp_:
        with open(qa_dataset_processed_path, 'wt', encoding='utf-8') as out_:
            for line in inp_.readlines():
                num_lines = num_lines+1
                try:
                    qa = line.strip().split('\t')
                    if len(qa) > 1:
                        num_lines_with_ans = num_lines_with_ans+1
                        answers = [i.strip() for i in qa[1].split('|')]
                        if(len(answers) == 1):
                            num_lines_with_one_ans = num_lines_with_one_ans+1
                            out_.write(line)
                except RuntimeError:
                    continue
    print(num_lines, ", ", num_lines_with_ans, ", ", num_lines_with_one_ans)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--kg_dataset", type=str, default="MetaQA", nargs="?",
                    help="Which dataset to use: MetaQA or WebQuestionsSP-237.")
    parser.add_argument("--hops", type=int, default="1", nargs="?",
                    help="Number of hops.")

    args = parser.parse_args()   
    KG_NAME = args.kg_dataset
    HOPS = args.hops
    KG_HALF = False

    if KG_NAME == 'MetaQA':
        QA_PAIRS_PATH = f'../QA/MetaQA/qa_%s_{str(HOPS)}hop{"_half" if KG_HALF else ""}.txt'
        QA_PAIRS_PROCESSED_PATH = f'../QA/Preprocess/MetaQA/qa_%s_{str(HOPS)}hop{"_half" if KG_HALF else ""}.txt'
        if not os.path.exists("../QA/Preprocess/MetaQA/"):
            os.makedirs("../QA/Preprocess/MetaQA/")
    else:
        QA_PAIRS_PATH = f'../QA/WebQuestionsSP/qa_%s_webqsp.txt'
        QA_PAIRS_PROCESSED_PATH = f'../QA/Preprocess/WebQuestionsSP/qa_%s_webqsp.txt'
        if not os.path.exists("../QA/Preprocess/WebQuestionsSP/"):
            os.makedirs("../QA/Preprocess/WebQuestionsSP/")

    print("QA_PAIRS_PATH - ", QA_PAIRS_PATH)
    print("QA_PAIRS_PROCESSED_PATH - ", QA_PAIRS_PROCESSED_PATH)
    qa_traindataset_path = QA_PAIRS_PATH % 'train'
    qa_traindataset_preprocess_path = QA_PAIRS_PROCESSED_PATH % 'train'
    qa_testdataset_path = QA_PAIRS_PATH % 'test'
    qa_testdataset_preprocess_path = QA_PAIRS_PROCESSED_PATH % 'test'

    if KG_NAME == 'MetaQA':
        qa_devdataset_path = QA_PAIRS_PATH % 'dev'
        qa_devdataset_preprocess_path = QA_PAIRS_PROCESSED_PATH % 'dev'
        remove_multiple_answers(qa_traindataset_path, qa_traindataset_preprocess_path)
        remove_multiple_answers(qa_devdataset_path, qa_devdataset_preprocess_path)
        remove_multiple_answers(qa_testdataset_path, qa_testdataset_preprocess_path)
    else:
        remove_multiple_answers(qa_traindataset_path, qa_traindataset_preprocess_path)
        remove_multiple_answers(qa_testdataset_path, qa_testdataset_preprocess_path)