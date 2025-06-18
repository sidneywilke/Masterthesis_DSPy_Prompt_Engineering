from datasets import load_dataset
from typing import Dict, Any, List
import dspy
import logging
import sys
from dotenv import load_dotenv

from dspy import LabeledFewShot, KNNFewShot, COPRO
import litellm
litellm.drop_params = True

load_dotenv()

#mlflow.set_tracking_uri("http://localhost:5000")
#mlflow.set_experiment("DSPy")
#mlflow.dspy.autolog()

# Logging the terminal outputs to a txt file to evaluate at as later stage
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

# Define a formatter for consistent log formatting
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Set up a file handler
file_handler = logging.FileHandler('output_neu_final_tiny5.log', mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Set up a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)

# Add both handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


# Custom class to redirect sys.stdout and sys.stderr to the logger
class LoggerWriter:
    def __init__(self, log_method):
        self.log_method = log_method
        self.buffer = ''

    def write(self, message):
        self.buffer += message
        if "\n" in self.buffer:
            self.flush()

    def flush(self):
        for line in self.buffer.rstrip().splitlines():
            try:
                self.log_method(line.rstrip())
            except UnicodeEncodeError:
                # Optionally, you could replace problematic characters:
                self.log_method(line.rstrip().encode('utf-8', errors='replace').decode('utf-8'))
        self.buffer = ''


# Redirect standard output and error to the logger.
sys.stdout = LoggerWriter(logger.info)
sys.stderr = LoggerWriter(logger.error)


def extract_people_entities(data_row: Dict[str, Any]) -> List[str]:
    """
    Extracts entities referring to people from a row of the CoNLL-2003 dataset.

    Args:
        data_row (Dict[str, Any]): A row from the dataset containing tokens and NER tags.

    Returns:
        List[str]: List of tokens tagged as people.
    """
    return [
        token
        for token, ner_tag in zip(data_row["tokens"], data_row["ner_tags"])
        if ner_tag in (1, 2)  # CoNLL entity codes 1 and 2 refer to people
    ]


def prepare_dataset(data_split, start: int, end: int) -> List[dspy.Example]:
    """
    Prepares a sliced dataset split for use with DSPy.

    Args:
        data_split: The dataset split (e.g., train or test).
        start (int): Starting index of the slice.
        end (int): Ending index of the slice.

    Returns:
        List[dspy.Example]: List of DSPy Examples with tokens and expected labels.
    """
    return [
        dspy.Example(
            tokens=row["tokens"],
            expected_extracted_people=extract_people_entities(row)
        ).with_inputs("tokens")
        for row in data_split.select(range(start, end))
    ]

# Load the dataset
dataset = load_dataset("conll2003", trust_remote_code=True)

# Prepare the training (max. 1000 items) and test sets
train_set = prepare_dataset(dataset["train"], 0, 200)
test_set = prepare_dataset(dataset["test"], 1500, 2500)





class PeopleExtraction(dspy.Signature):
    """
    Extract contiguous tokens referring to specific people, if any, from a list of string tokens.
    Output a list of tokens. In other words, do not combine multiple tokens into a single value.
    """
    tokens: list[str] = dspy.InputField(desc="tokenized text")
    extracted_people: list[str] = dspy.OutputField(desc="all tokens referring to specific people extracted from the tokenized text")


import os

llm= os.getenv("TINY_LLM")
api=os.getenv("LOCAL_API")

print("---------------------------------")
print(llm+"  "+api+" Runde 1")

people_extractor = dspy.ChainOfThought(PeopleExtraction)
lm = dspy.LM(llm, api_base=api, api_key='', cache=False)
dspy.configure(lm=lm)



def precision(tp: int, fp: int) -> float:
    # Handle division by zero
    return 0.0 if tp + fp == 0 else tp / (tp + fp)


def recall(tp: int, fn: int) -> float:
    return 0.0 if tp + fn == 0 else tp / (tp + fn)


def f1_score(tp: int, fp: int, fn: int) -> float:
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    return 0.0 if prec + rec == 0 else 2 * (prec * rec) / (prec + rec)



def extraction_correctness_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """
    Computes correctness of entity extraction predictions.

    Args:
        example (dspy.Example): The dataset example containing expected people entities.
        prediction (dspy.Prediction): The prediction from the DSPy people extraction program.
        trace: Optional trace object for debugging.

    Returns:
        bool: True if predictions match expectations, False otherwise.
    """

    gold_aspects = set(example.expected_extracted_people)
    pred_aspects = set(prediction.extracted_people)

    tp = +len(gold_aspects & pred_aspects)
    fp = +len(pred_aspects - gold_aspects)
    fn = +len(gold_aspects - pred_aspects)

    if len(gold_aspects) == 0 and len(pred_aspects) == 0:
        tp += 1  # correct prediction of no aspects



    return f1_score(tp, fp, fn)



evaluate_correctness = dspy.Evaluate(
    devset=test_set,
    metric=extraction_correctness_metric,
    num_threads=24,
    display_progress=True,
    display_table=True,
    return_outputs=True
)

print("---------------------------------------------------")
print("-----------------------Baseline--------------------")


evaluate_correctness(people_extractor, devset=test_set, return_outputs=True)


print("---------------------------------------------------")
print("-----------------------LabeledFewShot--------------")


fewshot_optimzer = LabeledFewShot(k=4)


fewshot_optimzer_compile = fewshot_optimzer.compile(people_extractor, trainset=train_set)

evaluate_correctness(fewshot_optimzer_compile, devset=test_set)

dspy.inspect_history(n=1)



print("---------------------------------------------------")
print("-----------------------KNNFewShot------------------")
from sentence_transformers import SentenceTransformer


knn_fewshot_optimizer = KNNFewShot(
    k=4,
    trainset=train_set,
    vectorizer=dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode)
)


knn_fewshot_optimizer_compile = knn_fewshot_optimizer.compile(people_extractor)


evaluate_correctness(knn_fewshot_optimizer_compile, devset=test_set)

dspy.inspect_history(n=1)



print("---------------------------------------------------")
print("-----------------------MIPROv2----------------------")

mipro_optimizer = dspy.MIPROv2(
    metric=extraction_correctness_metric,
    max_labeled_demos=4,
    max_bootstrapped_demos=4,
    num_candidates= 4,
    num_threads=8,
    init_temperature=1.2,
)

mipro_optimized_people_extractor = mipro_optimizer.compile(
    people_extractor,
    trainset=train_set,
    num_trials= 4,
    minibatch= True,
    requires_permission_to_run=False,
)


evaluate_correctness(mipro_optimized_people_extractor, devset=test_set)
dspy.inspect_history(n=1)



print("---------------------------------------------------")
print("-----------------------COPRO----------------------")


copro_optimizer = COPRO(
    prompt_model=lm,
    metric=extraction_correctness_metric,
    breadth=5,
    depth=4,
    init_temperature=1.2,
    track_stats=True,
)


with dspy.context(num_completions=1):
    copro_optimizer_compiled = copro_optimizer.compile(
        student=people_extractor,
        trainset=train_set,

        eval_kwargs=dict(num_threads=8)
)


evaluate_correctness(copro_optimizer_compiled, devset=test_set)
dspy.inspect_history(n=1)



print("---------------------------------------------------")
print("-----------------------BootstrapFewShot------------")


from dspy.teleprompt import BootstrapFewShot
bootstrap_fewshot_optimizer= BootstrapFewShot(
    metric=extraction_correctness_metric,
    max_bootstrapped_demos=4,
    max_labeled_demos=4,
    max_rounds=1,
    max_errors=5,
    teacher_settings=dict(lm=lm)
)

bootstrap_fewshot_optimizer_compiled=bootstrap_fewshot_optimizer.compile(people_extractor, trainset=train_set)


evaluate_correctness(bootstrap_fewshot_optimizer_compiled, devset=test_set)


dspy.inspect_history(n=1)



print("---------------------------------------------------")
print("-------BootstrapFewShotWithRandomSearch------------")

#BootstrapFewShot optimization
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
bootstrap_fewshot_withrandom_optimizer = BootstrapFewShotWithRandomSearch(
    metric=extraction_correctness_metric,
    max_bootstrapped_demos=4,
    num_candidate_programs=8,
    num_threads=8)

bootstrap_fewshot_withrandom_optimizer_compiled=bootstrap_fewshot_withrandom_optimizer.compile(people_extractor, trainset=train_set)


evaluate_correctness(bootstrap_fewshot_withrandom_optimizer_compiled, devset=test_set)


dspy.inspect_history(n=1)


