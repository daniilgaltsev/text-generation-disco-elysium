import torch
from typing import Optional, List
import os


class LinesDataset(torch.utils.data.Dataset):
    """
    A dataset class that loads and stores lines as strings.

    Args:
        path (optional): A path to the file containing script lines. 
            If None, will read texts_extracted.txt file in the folder where script_datasets.py file is located.
        bos_token (optional): A string to use a beginning of sequence token.
        eos_token (optional): A string to use a end of sequence token.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        bos_token: str = "<|BOS|>",
        eos_token: str = "<|EOS|>",
    ):
        super().__init__()

        if path is None:
            path = os.path.join(os.path.dirname(__file__), "texts_extracted.txt")
        self.bos_token = bos_token
        self.eos_token = eos_token

        script = self._load_script(path)
        self.lines = self._get_lines(script)


    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> str:
        return self.bos_token + self.lines[idx] + self.eos_token
    

    @staticmethod
    def _get_lines(script: str) -> List[str]:
        """
        Generates a list of lines from the given script.

        Args:
            script: A string containg the script.

        Returns:
            A list of script lines.
        """

        return script.split('\n')

    @staticmethod
    def _load_script(path: str) -> str:
        """
        Reads the script from the specified path.

        Args:
            path: path to the script file.

        Returns:
            A string with the contents of the script file.
        """

        fd = open(path)
        script = fd.read()
        fd.close()
        return script
