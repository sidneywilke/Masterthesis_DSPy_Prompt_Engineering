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
file_handler = logging.FileHandler('output_neu_final_sentiment.log', mode='a', encoding='utf-8')
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



import os

llm= "ollama_chat/gemma3:4b"
api=os.getenv("LOCAL_API")

print("---------------------------------")
print(llm+"  "+api+" Runde 1")
lm = dspy.LM(llm, api_base=api, api_key='', cache=False)
dspy.configure(lm=lm)
sentence = "it's a charming and often affecting journey."  # example from the SST-2 dataset.

# 1) Declare with a signature.
classify = dspy.Predict('sentence -> sentiment: bool')

# 2) Call with input argument(s).
response = classify(sentence=sentence)

# 3) Access the output.
print(response.sentiment)
print(response)



dspy.inspect_history(n=1)


