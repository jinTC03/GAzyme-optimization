from typing import Dict, Any, List, Tuple
import random

class NewSelectionGenerator:

    def top(self, pool: List[Dict[str, Any]], k: float) -> List[Dict[str, Any]]:
        #Select the individuals with highest fitness
        n_samples_to_select = int(len(pool) * k)
        return list(sorted(pool, key=lambda d: d["score"], reverse=True))[
            :n_samples_to_select
        ]


    def tournament(self, pool: List[Dict[str, Any]], k: float) -> List[Dict[str, Any]]:
        #Random select n and then select top p
        n_samples_to_select = int(len(pool) * k)
        selected_sequences = []
        n = 0.5
        top_fraction = 0.5
        tournament_size = max(2, int(len(pool) * n))
        while len(selected_sequences) < n_samples_to_select:
            select_n = random.sample(pool, min(tournament_size, len(pool)))
            count = max(1,int(len(select_n) * top_fraction))
            top_p = sorted(select_n, key=lambda d: d["score"], reverse=True)[:count]
            for p in top_p:
                if p["sequence"] not in selected_sequences:
                    selected_sequences.extend(top_p)

        return selected_sequences[:n_samples_to_select]


    def roulette(self, pool: List[Dict[str, Any]], k: float) -> List[Dict[str, Any]]:
        #Select individuals with probability proportional to their fitness
        total_score = 0
        n_samples_to_select = max(1, int(len(pool) * k))
        total_score = sum(i["score"] for i in pool)
        prob = [i["score"] / total_score for i in pool]
        selected_sequences = random.choices(pool, weights=prob, k=n_samples_to_select)
        return selected_sequences


    def selection(self, pool: List[Dict[str, Any]], k: float, method: str) -> List[Dict[str, Any]]:
        if method == 'top':
            return self.top(pool, k)
        elif method == 'tournament':
            return self.tournament(pool, k)
        elif method == 'roulette':
            return self.roulette(pool, k)



class NewCrossoverGenerator:

    def __init__(self, threshold_probability: float = 0.5) -> None:
        self.threshold_probability = threshold_probability


    def sp(self, a_sequence: str, another_sequence: str) -> Tuple[str, str]:
        #Exchange segments at one crossover point
        random_point = random.randint(1, len(a_sequence) - 2)
        print(a_sequence[:random_point] + another_sequence[random_point:],
            another_sequence[:random_point] + a_sequence[random_point:],
        )
        return (
            a_sequence[:random_point] + another_sequence[random_point:],
            another_sequence[:random_point] + a_sequence[random_point:],
        )


    def tp(
            self,a_sequence: str, another_sequence: str
    )-> Tuple[str, str]:
            #Exchange segments at two crossover point
            p1 = random.randint(1, len(a_sequence) - 3)
            p2 = random.randint(p1 + 1, len(a_sequence) - 1)
            return (
                a_sequence[:p1] + another_sequence[p1:p2] + a_sequence[p2:],
                another_sequence[:p1] + a_sequence[p1:p2] + another_sequence[p2:]
            )


    def uniform(
        self, a_sequence: str, another_sequence: str
    ) -> Tuple[str, str]:
        #Exchange each position with 50% probability
        return (
            "".join(
                a if random.random() > self.threshold_probability else b
                for a, b in zip(a_sequence, another_sequence)
            ),
            "".join(
                b if random.random() > self.threshold_probability else a
                for a, b in zip(a_sequence, another_sequence)
            ),
        )


    def crossover(self, a_sequence: str, another_sequence: str,intervals, method, use_intervals=True) -> Tuple[str, str]:
        if not use_intervals:
            intervals = [(0, len(a_sequence))]

        for start, end in intervals:
            a_sequence_in_interval = a_sequence[start:end]
            another_sequence_in_interval = another_sequence[start:end]

            if method == "sp":
                a_crossed, another_crossed = self.sp(a_sequence_in_interval, another_sequence_in_interval)
            elif method == "tp":
                a_crossed, another_crossed = self.tp(a_sequence_in_interval, another_sequence_in_interval)
            elif method == "uniform":
                a_crossed, another_crossed = self.uniform(a_sequence_in_interval, another_sequence_in_interval)

            a_sequence = a_sequence[:start] + a_crossed + a_sequence[end:]
            another_sequence = another_sequence[:start] + another_crossed + another_sequence[end:]

        return a_sequence, another_sequence





