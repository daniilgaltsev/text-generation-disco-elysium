from text_generation.models import GPT2LMHeadModel
import text_generation.utils
from typing import List, Union, Optional


class TextGenerator:
    """
    Given a number generates this number of sequences.

    Args:
        load_path (optional): A path from which to load the model and the tokenizer.
        model_cls (optional): A class which will be used to create a model.
    """

    def __init__(self, load_path: str = text_generation.utils.WEIGHTS_PATH, model_cls=GPT2LMHeadModel):
        self.tokenizer = text_generation.utils.get_tokenizer(load_path)
        self.model = model_cls(self.tokenizer, load_path)


    def generate(
        self,
        n_sequences: Union[int, List[str]] = 1,
        max_token_length: int = 80,
        max_char_length: Optional[int] = 270
        ) -> List[str]:
        """
        Generates 'n_sequences' of sequences.

        Args:
            n_sequences (optional): A number of sequences to generate or a list of prompts to generate from.
            max_token_length (optional): The maximum length of generated sequences in tokens.
            max_char_length (optional): The maximum length of generated sequences in characters. If None, does not truncate.
        """

        if isinstance(n_sequences, int):
            n_sequences = [""] * n_sequences

        generated = self.model.generate(n_sequences, max_length=max_token_length)

        for idx, seq in enumerate(generated):
            generated[idx] = seq[:max_char_length]

        return generated