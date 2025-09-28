# code borrowed from github repository https://github.com/GT4SD/enzeptional.git

import warnings

from enzeptional import (
    HuggingFaceEmbedder,
    HuggingFaceTokenizerLoader,
)

from typing import List 

warnings.simplefilter(action="ignore", category=FutureWarning)

class ProteinLLM:

    """
    Wrapper for the GT4SD-based protein language model for sequence mutation generation.
    """

    def __init__(self, model_loader: str, tokenizer_loader: str, language_model_path: str, tokenizer_path: str, device = 'cpu'):
        
        """
        Initialize the ProteinLLM generator.

        Args:
            model_loader: The loader responsible for loading the Hugging Face model.
            tokenizer_loader: The loader responsible for loading the Hugging Face tokenizer.
            model_path: The path to the Hugging Face model.
            tokenizer_path: The path to the Hugging Face tokenizer.
            cache_dir: Optional directory where the model is cached.
            device: The device on which to load the model (e.g., 'cpu' or 'cuda').
        """

        self.generator = HuggingFaceEmbedder(model_loader=model_loader, tokenizer_loader=tokenizer_loader, model_path=language_model_path,
                        tokenizer_path=tokenizer_path,  cache_dir=None, device = device)
        
        self.protein_tokenizer = HuggingFaceTokenizerLoader()
        

class ChemicalLLM:

    """
    Wrapper for the GT4SD class for embedding generation from ChemBERTa model.

    """

    def __init__(self, model_loader, tokenizer_loader, chem_model_path: str, chem_tokenizer_path: str, device: str):
        
        """
        Initialize the ChemBERTa.

        Args:
            model_loader: The loader responsible for loading the Hugging Face model.
            tokenizer_loader: The loader responsible for loading the Hugging Face tokenizer.
            chem_model_path: The path to the Hugging Face model.
            chem_tokenizer_path: The path to the Hugging Face tokenizer.
            cache_dir: Optional directory where the model is cached.
            device: The device on which to load the model (e.g., 'cpu' or 'cuda').
        """

        self.chem_model = HuggingFaceEmbedder(model_loader=model_loader, tokenizer_loader=tokenizer_loader, model_path=chem_model_path,
                        tokenizer_path=chem_tokenizer_path,  cache_dir=None, device=device)
        
    def embed(self, samples: List[str]):
        return self.chem_model.embed(samples)
