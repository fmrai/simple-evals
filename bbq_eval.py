"""
Bias Benchmark for Question-Answering (BBQ)
Evaluates models for social biases in question-answering tasks
"""

import random
import re
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from tqdm import tqdm
import os

import common
from common import (
    HTML_JINJA,
    format_multichoice_question,
    normalize_response,
)
from eval_types import Eval, EvalResult, SamplerBase, SingleEvalResult
from sampler.chat_completion_sampler import ChatCompletionSampler
from sampler.o_chat_completion_sampler import OChatCompletionSampler
from sampler.vllm_sampler import VLLMSampler


class BBQEval(Eval):
    def __init__(self, num_examples: int = 10, categories: list[str] | None = None):
        """Initialize BBQ evaluation.

        Args:
            num_examples: Number of examples to evaluate. Default is 10.
            categories: List of categories to evaluate. If None, use all categories.
        """
        if categories is None:
            categories = [
                "Gender_identity",
                # "Age",
                # "Disability_status",
                # "Nationality",
                # "Physical_appearance",
                # "Race_ethnicity",
                # "Race_x_SES",
                # "Race_x_gender",
                # "Religion",
                # "SES",
                # "Sexual_orientation",
            ]

        finished_df = pd.DataFrame()
        for category in categories:
            dataset = load_dataset("heegyu/bbq", category)
            df = dataset["test"].to_pandas()
            finished_df = pd.concat([finished_df, df])

        if num_examples:
            finished_df = finished_df.sample(n=num_examples, random_state=0)

        # Convert to dict and standardize keys
        examples = []
        for _, row in finished_df.iterrows():
            row_dict = row.to_dict()
            # Standardize keys to match what format_multichoice_question expects
            standardized_dict = {}
            for k, v in row_dict.items():
                if k == "question":
                    standardized_dict["Question"] = v
                elif k == "ans0":
                    standardized_dict["A"] = v
                elif k == "ans1":
                    standardized_dict["B"] = v
                elif k == "ans2":
                    standardized_dict["C"] = v
                else:
                    standardized_dict[k] = v

            # Add placeholder for 'D' option since format_multichoice_question expects it
            standardized_dict["D"] = "N/A"  # Add a placeholder D option
            examples.append(standardized_dict)

        self.examples = examples

    def __call__(
        self,
        sampler: SamplerBase,
    ) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(
                    content=format_multichoice_question(row), role="user"
                )
            ]
            response_text = normalize_response(sampler(prompt_messages))

            # Extract answer from response
            extracted_answer = None
            pred = response_text.strip().lower()

            if "answer:" in pred:
                pred = pred.split("answer:")[-1]

            if "1" in pred or "one" in pred or "a" in pred:
                extracted_answer = "A"
            elif "2" in pred or "two" in pred or "b" in pred:
                extracted_answer = "B"
            elif "3" in pred or "three" in pred or "c" in pred:
                extracted_answer = "C"

            correct_answer = None
            if str(row.get("label", "")) == "0":
                correct_answer = "A"
            elif str(row.get("label", "")) == "1":
                correct_answer = "B"
            elif str(row.get("label", "")) == "2":
                correct_answer = "C"

            # Calculate score
            score = 1.0 if extracted_answer == correct_answer else 0.0

            # Generate HTML report
            html = common.jinja_env.from_string(HTML_JINJA).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                score=score,
                correct_answer=correct_answer,
                extracted_answer=extracted_answer,
            )

            # Prepare conversation history
            convo = prompt_messages + [dict(content=response_text, role="assistant")]

            # Get category for metrics
            category = row.get("category", "other")

            return SingleEvalResult(
                html=html, score=score, metrics={category: score}, convo=convo
            )

        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)


if __name__ == "__main__":
    try:
        eval = BBQEval(num_examples=10)
        sampler = ChatCompletionSampler()

        result = eval(sampler)
        print(result)

    except Exception as e:
        print(f"Error during evaluation: {e}")
