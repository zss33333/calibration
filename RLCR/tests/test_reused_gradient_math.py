"""Check the cross-rank and cross-microbatch terms in gradient projection."""

import math
import unittest

from reused_gradient_math import training_to_raw_sum_scale


class ReusedGradientMathTest(unittest.TestCase):
    def test_sharded_projection_matches_independent_full_vectors(self):
        # Three microbatches on four ranks; varied signs make omitted cross
        # terms or incorrect rank averaging visible in the final projection.
        full = [
            [[0.4 + rank * 0.1 - micro * 0.07 + col * 0.03 for col in range(17)]
             for micro in range(3)]
            for rank in range(4)
        ]
        no_brier = [
            [[full[rank][micro][col] - (-1) ** (rank + micro + col) * (0.02 + col * 0.001)
              for col in range(17)] for micro in range(3)]
            for rank in range(4)
        ]
        full_global = [sum(full[r][m][j] for r in range(4) for m in range(3))
                       for j in range(17)]
        no_brier_global = [sum(no_brier[r][m][j] for r in range(4) for m in range(3))
                           for j in range(17)]
        reference_dot = sum(f * n for f, n in zip(full_global, no_brier_global))
        reference_norm = sum(n * n for n in no_brier_global)

        # Trainer and DeepSpeed each divide the scalar loss by GAS; ZeRO-2
        # averages the gradient across the data-parallel ranks.
        shard_len = math.ceil(17 / 4)
        sharded_dot = sharded_norm = 0.0
        for rank in range(4):
            lo, hi = rank * shard_len, min((rank + 1) * shard_len, 17)
            for j in range(lo, hi):
                averaged_train_grad = full_global[j] / (3 * 3 * 4)
                restored_full = averaged_train_grad * training_to_raw_sum_scale(3, 4)
                sharded_dot += restored_full * no_brier_global[j]
                sharded_norm += no_brier_global[j] * no_brier_global[j]
        self.assertAlmostEqual(sharded_dot / sharded_norm,
                               reference_dot / reference_norm, places=12)


if __name__ == "__main__":
    unittest.main()
