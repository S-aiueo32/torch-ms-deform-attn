"""Exercise CUDA index planning without a GPU or large tensor allocations."""

import unittest

from torch_ms_deform_attn import _C


class CUDAIndexingTest(unittest.TestCase):
    def test_value_offsets_exceed_int32(self):
        # The third batch starts at 2**31 elements in this formerly unsafe case.
        plan = _C._check_cuda_indexing([3, 4194304, 1, 256, 1, 1, 1], 3)
        self.assertEqual(plan, [3, 2**30, 2, 1, 768, 3, 0])
        self.assertEqual(2 * plan[1], 2**31)

    def test_location_offsets_exceed_int32(self):
        plan = _C._check_cuda_indexing([3, 1, 1, 1, 1, 2**29, 1], 3)
        self.assertEqual(plan[2], 2**30)
        self.assertEqual(plan[4], 3 * 2**29)
        self.assertEqual(plan[6], 0)

    def test_work_exceeds_int32(self):
        plan = _C._check_cuda_indexing([1, 1, 1, 1024, 1, 2**21 + 1, 1], 1)
        self.assertEqual(plan[4], 2**31 + 1024)
        self.assertEqual(plan[5], 2**21 + 1)
        self.assertEqual(plan[6], 0)

    def test_grid_limit_can_be_resolved_by_chunking(self):
        dims = [2, 1, 1, 1, 1, 2**31 - 1, 1]
        with self.assertRaisesRegex(RuntimeError, "grid dimension limit"):
            _C._check_cuda_indexing(dims, 2)
        plan = _C._check_cuda_indexing(dims, 1)
        self.assertEqual(plan[4:], [2**31 - 1, 2**31 - 1, 0])

    def test_partial_chunk_plan(self):
        self.assertEqual(
            _C._check_cuda_indexing([3, 16, 2, 71, 2, 2, 2], 2), [2, 2272, 32, 16, 568, 8, 1]
        )

    def test_int32_value_neighbor_headroom(self):
        # The output is tiny; only the value span can select the fallback.
        limit = (2**31 - 1) // 2
        for spatial_size, expected in ((limit, 1), (limit + 1, 0)):
            with self.subTest(spatial_size=spatial_size):
                plan = _C._check_cuda_indexing([1, spatial_size, 1, 1, 1, 1, 1], 1)
                self.assertEqual(plan[6], expected)

    def test_int32_locations_and_metadata_headroom(self):
        # Locations have an extra coordinate dimension that output does not.
        for queries, expected in ((2**29 - 1, 1), (2**29, 0)):
            with self.subTest(queries=queries):
                plan = _C._check_cuda_indexing([1, 1, 1, 1, 1, queries, 1], 1)
                self.assertEqual(plan[6], expected)
        # The same bound protects 2 * levels when the output remains tiny.
        for levels, expected in ((2**29 - 1, 1), (2**29, 0)):
            with self.subTest(levels=levels):
                plan = _C._check_cuda_indexing([1, 1, 1, 1, levels, 1, 1], 1)
                self.assertEqual(plan[6], expected)

    def test_int32_padded_grid_increment_headroom(self):
        # A kernel's final loop increment must fit even for padded grid lanes.
        for queries, expected in ((2**20 - 1, 1), (2**20, 0)):
            with self.subTest(queries=queries):
                plan = _C._check_cuda_indexing([1, 1, 1, 1024, 1, queries, 1], 1)
                self.assertEqual(plan[6], expected)

    def test_int32_selection_is_chunk_local(self):
        # Host offsets exceed int32, while each device launch remains bounded.
        dims = [8, 2**28, 1, 1, 1, 1, 1]
        self.assertEqual(_C._check_cuda_indexing(dims, 1)[6], 1)
        self.assertEqual(_C._check_cuda_indexing(dims, 8)[6], 0)
        # Chunk-local sampling offsets need the same treatment as values.
        dims = [8, 1, 1, 1, 1, 2**27, 1]
        self.assertEqual(_C._check_cuda_indexing(dims, 1)[6], 1)
        self.assertEqual(_C._check_cuda_indexing(dims, 8)[6], 0)

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
