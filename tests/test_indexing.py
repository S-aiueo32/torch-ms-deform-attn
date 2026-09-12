"""Exercise CUDA index planning without a GPU or large tensor allocations."""
import unittest

from torch_deform_attn import _C


class CUDAIndexingTest(unittest.TestCase):
    def test_value_offsets_exceed_int32(self):
        # The third batch starts at 2**31 elements in this formerly unsafe case.
        plan = _C._check_cuda_indexing([3, 4194304, 1, 256, 1, 1, 1], 3)
        self.assertEqual(plan, [3, 2**30, 2, 1, 768, 3])
        self.assertEqual(2 * plan[1], 2**31)

    def test_location_offsets_exceed_int32(self):
        plan = _C._check_cuda_indexing([3, 1, 1, 1, 1, 2**29, 1], 3)
        self.assertEqual(plan[2], 2**30)
        self.assertEqual(plan[4], 3 * 2**29)

    def test_work_exceeds_int32(self):
        plan = _C._check_cuda_indexing([1, 1, 1, 1024, 1, 2**21 + 1, 1], 1)
        self.assertEqual(plan[4], 2**31 + 1024)
        self.assertEqual(plan[5], 2**21 + 1)

    def test_grid_limit_can_be_resolved_by_chunking(self):
        dims = [2, 1, 1, 1, 1, 2**31 - 1, 1]
        with self.assertRaisesRegex(RuntimeError, "grid dimension limit"):
            _C._check_cuda_indexing(dims, 2)
        plan = _C._check_cuda_indexing(dims, 1)
        self.assertEqual(plan[4:], [2**31 - 1, 2**31 - 1])

    def test_partial_chunk_plan(self):
        self.assertEqual(_C._check_cuda_indexing([3, 16, 2, 71, 2, 2, 2], 2),
                         [2, 2272, 32, 16, 568, 8])

    def test_invalid_dimensions_and_overflow(self):
        for index in range(7):
            for invalid in (0, -1, 2**31):
                dims = [1] * 7
                dims[index] = invalid
                with self.subTest(index=index, invalid=invalid):
                    with self.assertRaisesRegex(RuntimeError, "dimensions.*int32"):
                        _C._check_cuda_indexing(dims, 1)
        for step in (0, -1, 2**31):
            with self.assertRaisesRegex(RuntimeError, "im2col_step"):
                _C._check_cuda_indexing([1] * 7, step)
        with self.assertRaisesRegex(RuntimeError, "64-bit indexing"):
            _C._check_cuda_indexing([2**31 - 1] * 7, 1)
        with self.assertRaisesRegex(RuntimeError, "seven dimensions"):
            _C._check_cuda_indexing([1] * 6, 1)


if __name__ == "__main__":
    unittest.main()
