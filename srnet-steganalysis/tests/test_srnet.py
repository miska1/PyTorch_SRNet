from __future__ import annotations

import unittest

import torch

from src.srnet import SRNet


class SRNetTests(unittest.TestCase):
    def test_output_shape(self) -> None:
        model = SRNet().eval()
        with torch.no_grad():
            output = model(torch.zeros(2, 1, 32, 32))
        self.assertEqual(output.shape, (2, 2))


if __name__ == "__main__":
    unittest.main()
