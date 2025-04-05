from .cot import CoTModel
import torch
from .data import Dataset, is_answer_valid

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
def generate_dataset(output_json: str, oversample: int = 10, temperature: float = 0.6):
    # questions = [["Can you change 2 hour to its equivalent in min?", 120.0], ["Express 4 centuries as a quantity of week.", 20870.982787499997]]
    questions = ["Can you change 2 hour to its equivalent in min?","Express 4 centuries as a quantity of week." ]
    model = CoTModel(checkpoint=checkpoint)

    train_raw = Dataset("train")
    num_records = len(train_raw)

    idx = range(num_records)
    # questions = [train_raw[i][0] for i in idx]
    print(f"number of question: {len(questions)}")

    prompts = [model.format_prompt(q) for q in questions]
    generations = model.batched_generate(prompts, num_return_sequences=oversample, temperature=temperature)

    print(generations)
    for i in range(len(questions)):
        for j in range(oversample):
            generation = generations[i][j]
            generated_answer = model.parse_answer(generation)
            groundtruth = train_raw[i][1]
            print(f"generation: {generation}")
            print(f"generated answer: {generated_answer}; groundtruth: {groundtruth}")
            print("========")
            # if is_answer_valid(generated_answer, groundtruth):




if __name__ == "__main__":
    from fire import Fire

    Fire(generate_dataset)
