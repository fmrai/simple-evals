"""
Bias Benchmark for Question-Answering (BBQ)
Evaluates models for social biases in question-answering tasks
"""

import random
import pandas as pd

import common
from common import (
    normalize_response,
)
from eval_types import Eval, EvalResult, SamplerBase, SingleEvalResult
from sampler.chat_completion_sampler import ChatCompletionSampler


HTML_JINJA_POLYGRAPH = """
<h3>Prompt conversation</h3>
{% for message in prompt_messages %}
{{ message_to_html(message) | safe }}
{% endfor %}
<h3>Sampled message</h3>
{{ message_to_html(next_message) | safe }}
<h3>Results</h3>
<p>Question: {{ question }}</p>
<p>Answer: {{ answer }}</p>
<p>Score: {{ score }}</p>
"""

def read_dqa(dataset_path: str):
    df = pd.read_csv(dataset_path)
    return df

class PolygraphEval(Eval):
    def __init__(self, dataset_path: str, num_examples: int):    
        self.dataset_path = dataset_path
        df = read_dqa(self.dataset_path)

        examples = [row.to_dict() for _, row in df.iterrows()]
        if num_examples:
            examples = random.Random(0).sample(examples, num_examples)
        self.examples = examples

    def __call__(
        self,
        sampler: SamplerBase,
    ) -> EvalResult:
        def fn(row: dict):
            prompt_messages = [
                sampler._pack_message(
                    content=row["q"], role="user"
                ),
                sampler._pack_message(
                    content=row["y"], role="assistant"
                ),
            ]
            response_text = normalize_response(sampler(prompt_messages, max_tokens=1))

            score = self.get_score(row)

            # Generate HTML report
            html = common.jinja_env.from_string(HTML_JINJA_POLYGRAPH).render(
                prompt_messages=prompt_messages,
                next_message=dict(content=response_text, role="assistant"),
                question=row["q"],
                answer=row["y"],
                score=score,
            )

            # Prepare conversation history
            convo = prompt_messages + [dict(content=response_text, role="assistant")]

            return SingleEvalResult(
                html=html, score=score, convo=convo
            )

        print(self.examples, len(self.examples))
        results = common.map_with_progress(fn, self.examples)
        return common.aggregate_results(results)
    
    def get_score(self, row: dict) -> float:

        # TODO: parse the log file to get the score
        return 0.0


if __name__ == "__main__":
    try:
        eval = PolygraphEval(num_examples=10)
        sampler = ChatCompletionSampler()

        result = eval(sampler)
        print(result)

    except Exception as e:
        print(f"Error during evaluation: {e}")
