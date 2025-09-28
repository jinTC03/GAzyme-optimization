
from typing import Dict, Any, List, Tuple
from enzeptional.core import SequenceMutator

class SequenceMutatorInitializer:

    """
    Wrapper for the GT4SD package SequenceMutator class. Intializes with sequence and mutation strategy  

    """

    def __init__(self, sequence: str, mutation_config: Dict[str, Any]):

        """Initializes the mutator with a sequence and a mutation strategy.

        Args:
            sequence: The sequence to be mutated.
            mutation_config: Configuration for
            the mutation strategy.
        """

        self.mutator = SequenceMutator(sequence, mutation_config)
    
    def set_top_k(self, k):
            return self.mutator.set_top_k(k)

    def mutate_sequences(
            self,
            num_sequences: int,
            number_of_mutations: int,
            intervals: List[Tuple[int, int]],
            current_population: List[str],
            all_mutated_sequences: set[str],
    ) -> List[str]:
        return self.mutator.mutate_sequences(
            num_sequences=num_sequences,
            number_of_mutations=number_of_mutations,
            intervals=intervals,
            current_population=current_population,
            all_mutated_sequences=all_mutated_sequences
        )