import unittest

import torch

from src.losses import compute_loss, mask_to_edge
from src.metrics import segmentation_metrics
from src.models import build_model


class CoreTests(unittest.TestCase):
    def test_model_shapes(self):
        x = torch.randn(2, 3, 64, 64)
        baseline = build_model("baseline", base=8)(x)
        improved = build_model("improved", base=8)(x)
        self.assertEqual(tuple(baseline["mask"].shape), (2, 1, 64, 64))
        self.assertEqual(tuple(improved["mask"].shape), (2, 1, 64, 64))
        self.assertEqual(tuple(improved["edge"].shape), (2, 1, 64, 64))

    def test_losses_are_finite(self):
        target = (torch.rand(2, 1, 32, 32) > 0.7).float()
        baseline_loss, _ = compute_loss({"mask": torch.randn_like(target)}, target, "baseline")
        improved_loss, _ = compute_loss(
            {"mask": torch.randn_like(target), "edge": torch.randn_like(target)}, target, "improved"
        )
        self.assertTrue(torch.isfinite(baseline_loss))
        self.assertTrue(torch.isfinite(improved_loss))
        self.assertEqual(mask_to_edge(target).shape, target.shape)

    def test_perfect_metrics(self):
        target = torch.zeros(1, 1, 16, 16)
        target[:, :, 4:12, 4:12] = 1
        logits = torch.where(target > 0, torch.tensor(20.0), torch.tensor(-20.0))
        metrics = segmentation_metrics(logits, target)
        self.assertGreater(metrics["iou"], 0.999)
        self.assertGreater(metrics["dice"], 0.999)


if __name__ == "__main__":
    unittest.main()

