from functools import lru_cache
import math

START = 32
TARGET = 4205

MIN_LAYERS = 4
MAX_LAYERS = 8

STRIDES = (2, 3, 4)
KERNELS = (3, 4, 5, 6)
PADDINGS = (0, 1, 2)


def convT_out(L, s, k, p, op):
    return (L - 1) * s - 2 * p + k + op


def previous_lengths(L_out):
    """
    Find all possible L_in values that can produce L_out
    """

    prev = []

    for s in STRIDES:
        for k in KERNELS:
            for p in PADDINGS:

                    numerator = L_out - k + 2*p

                    if numerator % s != 0:
                        continue

                    L_in = numerator // s + 1

                    if L_in <= 0:
                        continue

                    prev.append(
                        (
                            L_in,
                            (s, k, p)
                        )
                    )

    return prev


def architecture_score(lengths, params):
    """
    Lower is better.

    Penalizes:
    - uneven growth
    - too many layers
    - large kernels
    - output padding
    """

    growths = [
        lengths[i+1] / lengths[i]
        for i in range(len(lengths)-1)
    ]

    # smoothness of multiplicative growth
    log_growths = [
        math.log(g)
        for g in growths
    ]

    mean_growth = sum(log_growths) / len(log_growths)

    growth_variance = sum(
        (g - mean_growth)**2
        for g in log_growths
    )

    score = 0

    # main term: smooth scaling
    score += 100 * growth_variance

    # prefer fewer layers, but not excessively
    score += 2 * len(params)

    for s, k, p in params:

        # prefer kernels around 5
        score += 2 * abs(k - 5)

        # prefer stride 2/4, avoid stride 3
        if s == 3:
            score += 5

    return score


@lru_cache(None)
def search(length, remaining_layers):

    if remaining_layers == 0:

        if length == START:
            return [([], [START])]

        return []

    solutions = []

    for prev_length, param in previous_lengths(length):

        paths = search(
            prev_length,
            remaining_layers - 1
        )

        for params, lengths in paths:

            solutions.append(
                (
                    params + [param],
                    lengths + [length]
                )
            )

    return solutions

if __name__ == "__main__":
    best_solution = None
    best_score = float("inf")


    for n_layers in range(MIN_LAYERS, MAX_LAYERS + 1):

        solutions = search(
            TARGET,
            n_layers
        )

        print(
            f"{n_layers} layers: {len(solutions)} solutions"
        )

        for params, lengths in solutions:

            score = architecture_score(
                lengths,
                params
            )

            if score < best_score:

                best_score = score
                best_solution = (
                    lengths,
                    params
                )


    print("\nBEST SOLUTION")
    print("----------------")

    lengths, params = best_solution

    print(lengths)

    print()

    for i, (p, k, pad) in enumerate(params):

        print(
            f"Layer {i+1}: "
            f"{lengths[i]} -> {lengths[i+1]} | "
            f"kernel={k}, "
            f"stride={p}, "
            f"padding={pad}"
        )

    print()
    print("Score:", best_score)