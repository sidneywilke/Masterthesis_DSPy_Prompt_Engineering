import os
import tempfile
from datasets import load_dataset
from typing import Dict, Any, List
import dspy


import logging
import sys
import io

# Optionally change console encoding to UTF-8 (run in Windows cmd)
# Uncomment the next line if you face encoding issues in the console.
# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Create (or get) the logger and clear previous handlers if re-running in an interactive session
logger = logging.getLogger('my_logger')
logger.setLevel(logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()

# Define a formatter for consistent log formatting
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Set up a file handler with UTF-8 encoding
file_handler = logging.FileHandler('output.log', mode='a', encoding='utf-8')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)

# Set up a console handler (it will use sys.stdout which we've optionally set to UTF-8)
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




def load_conll_dataset() -> dict:
    """
    Loads the CoNLL-2003 dataset into train, validation, and test splits.

    Returns:
        dict: Dataset splits with keys 'train', 'validation', and 'test'.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        # Use a temporary Hugging Face cache directory for compatibility with certain hosted notebook
        # environments that don't support the default Hugging Face cache directory
        os.environ["HF_DATASETS_CACHE"] = temp_dir
        return load_dataset("conll2003", trust_remote_code=True)


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
dataset = load_conll_dataset()

# Prepare the training and test sets
train_set = prepare_dataset(dataset["train"], 300, 350)
test_set = prepare_dataset(dataset["test"], 400, 600)



from typing import List

class PeopleExtraction(dspy.Signature):
    """
    Extract contiguous tokens referring to specific people, if any, from a list of string tokens.
    Output a list of tokens. In other words, do not combine multiple tokens into a single value.
    """
    tokens: list[str] = dspy.InputField(desc="tokenized text")
    extracted_people: list[str] = dspy.OutputField(desc="all tokens referring to specific people extracted from the tokenized text")

people_extractor = dspy.ChainOfThought(PeopleExtraction)

lm = dspy.LM('ollama_chat/qwen2.5:14b', api_base='http://localhost:11434', api_key='')
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



#evaluate_correctness(people_extractor, devset=test_set, return_outputs=True)



'''mipro_optimizer = dspy.MIPROv2(
    metric=extraction_correctness_metric,
    auto="medium",
)
optimized_people_extractor = mipro_optimizer.compile(
    people_extractor,
    trainset=train_set,
    max_bootstrapped_demos=4,
    requires_permission_to_run=False,
    minibatch=False
)

evaluate_correctness(optimized_people_extractor, devset=test_set)
dspy.inspect_history(n=1)'''


from dspy.teleprompt import BootstrapFewShotWithOptuna
bootstrap_optimizer = BootstrapFewShotWithOptuna(
    metric=extraction_correctness_metric,
    max_bootstrapped_demos=2,
    num_candidate_programs=8,
)
bootstrap_optimizer_compiled=bootstrap_optimizer.compile(people_extractor, max_demos=4, trainset=train_set)

evaluate_correctness(bootstrap_optimizer_compiled, devset=test_set)
dspy.inspect_history(n=1)