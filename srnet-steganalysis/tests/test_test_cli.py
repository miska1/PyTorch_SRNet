from __future__ import annotations

import unittest

from src.test import build_parser, verify_checkpoint_test_set


class StandaloneTestCliTests(unittest.TestCase):
    def test_external_test_sets_are_allowed_by_default(self) -> None:
        args = build_parser().parse_args(["--checkpoint", "best.pt"])
        self.assertFalse(args.verify_checkpoint_test_set)

    def test_checkpoint_test_set_verification_is_opt_in(self) -> None:
        args = build_parser().parse_args(
            ["--checkpoint", "best.pt", "--verify-checkpoint-test-set"]
        )
        self.assertTrue(args.verify_checkpoint_test_set)

    def test_external_test_set_is_accepted_by_default(self) -> None:
        checkpoint = {"dataset_keys": {"test": ["original"]}}
        verify_checkpoint_test_set(checkpoint, ["external"], required=False)

    def test_strict_verification_rejects_a_different_test_set(self) -> None:
        checkpoint = {"dataset_keys": {"test": ["original"]}}
        with self.assertRaisesRegex(ValueError, "differs"):
            verify_checkpoint_test_set(checkpoint, ["external"], required=True)


if __name__ == "__main__":
    unittest.main()
