from .base_llm import BaseLLM
from .sft import test_model, TokenizedDataset
from .data import Dataset, benchmark
from peft import LoraConfig, TaskType, get_peft_model
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

def load() -> BaseLLM:
    from pathlib import Path

    from peft import PeftModel

    model_name = "rft_model"
    model_path = Path(__file__).parent / model_name

    llm = BaseLLM()
    llm.model = PeftModel.from_pretrained(llm.model, model_path).to(llm.device)
    llm.model.eval()

    return llm


def format_example(question: str, answer: str, reason: str) -> dict[str, str]:
    """
    Construct a question / answer pair. Consider rounding the answer to make it easier for the LLM.
    """

    return {
        "question": question,
        "answer": reason
    }

def train_model(
    output_dir: str,
    **kwargs,
):
    basellm = BaseLLM()
    base_model = basellm.model

    data = Dataset("rft")
    tokenized_dataset = TokenizedDataset(basellm.tokenizer, data, format_example)

    # get lora model
    lora_config = LoraConfig(
        target_modules="all-linear",
        bias="none",
        task_type="CAUSAL_LM",
        r=6,
        lora_alpha=16
    )

    peft_model = get_peft_model(base_model, lora_config)

    if basellm.device == "cuda":
        peft_model.enable_input_require_grads()

    # define training argument
    training_args = TrainingArguments(
        gradient_checkpointing=True,
        learning_rate=2e-3,
        output_dir=output_dir,
        logging_dir=output_dir,
        report_to="tensorboard",
        per_device_train_batch_size=32,
        num_train_epochs=7,
        logging_steps=50,
        push_to_hub=False,
    )

    # data collator
    data_collator = DataCollatorWithPadding(tokenizer=basellm.tokenizer)

    # instantiate trainer
    trainer = Trainer(
        model=peft_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=basellm.tokenizer,
        data_collator=data_collator,
    )

    print("start training")
    trainer.train()

    trainer.save_model("./homework/rft_model")

    test_model("./homework/rft_model")

if __name__ == "__main__":
    from fire import Fire

    Fire({"train": train_model, "test": test_model, "load": load})
