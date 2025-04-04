from .base_llm import BaseLLM


class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into a chat template. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        conversation = [
            {
                "role": "system",
                "content": (
                    "You will be given a task of unit conversion. Be very concise. Give your answer in "
                    " the format of <answer>{answer}</answer>."
                )
            },
            {
                "role": "user",
                "content": "What is the equivalence of 1 kilometer in meter?",
            },
            {
                "role": "assistant",
                "content": "<answer>1000</answer>"
            },
            {
                "role": "user",
                "content": question
            }
        ]

        formatted_question = self.tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
        return formatted_question

def load() -> CoTModel:
    return CoTModel()


def test_model():
    from .data import Dataset, benchmark

    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model, "load": load})
