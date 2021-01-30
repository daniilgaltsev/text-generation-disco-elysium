from typing import List, Dict, Any
import torch
import transformers


class GPT2LMHeadModel(torch.nn.Module):
    """
    Loads and contains gpt2 model with language modelling head.

    Args:
        tokenizer: A tokenizer to use for encoding given sequences.
        weights_path (optional): A path to the pre-trained model configuration or directory it's in.
    """

    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizerBase,
        weights_path: str = "gpt2",
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.model = transformers.GPT2LMHeadModel.from_pretrained(
            weights_path,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        self.model.resize_token_embeddings(len(tokenizer))

    def forward(self, x: Dict[str, torch.Tensor]) -> Any:
        return self.model(**x)

    def generate(
        self,
        prompts: List[str],
        max_length: int = 40,
        clean_special_tokens: bool = True
    ) -> List[str]:
        """
        Generates seqeunces given a list of prompts

        Args:
            promts: A list of string, which will be used as prompts for generation.
            max_length (optional): Maximum number of tokens to output (includes length of the prompt).
            clean_special_tokens (optional): If True will remove special tokens from generated strings.

        Returns:
            A list of generated strings corresponding to given prompts.
        """

        encoded = self.tokenizer(prompts, add_special_tokens=True, padding=True, return_tensors='pt')

        input_ids = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        if input_ids.size(1) == 0:
            input_ids = torch.ones(
                (input_ids.size(0), 1),
                dtype=torch.int64
            ) * self.tokenizer.bos_token_id
            attention_mask = torch.ones(
                (input_ids.size(0), 1),
                dtype=torch.int64
            )
        device = next(self.model.parameters()).device
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        output = self.model.generate(
            input_ids=input_ids, 
            attenttion_mask=attention_mask, 
            max_length=max_length, 
            do_sample=True, 
            num_return_sequences=1, 
            top_k=100, 
            top_p=0.99
        )
        output = self.tokenizer.batch_decode(
            output, 
            skip_special_tokens=clean_special_tokens, 
            clean_up_tokenization_spaces=clean_special_tokens
        )

        return output

    def save(self, save_directory: str) -> None:
        """
        Saves the model at specified directory.

        Args:
            save_directory: A directory where to save the model.
        """

        self.model.save_pretrained(save_directory)