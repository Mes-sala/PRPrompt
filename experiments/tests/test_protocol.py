"""Scientific contract tests on a tiny graph; no external data/checkpoints needed."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch
from prprompt.model import (TwoLayerGCN, normalized_adjacency, attribute_features,
                            prototype_route, PrototypeRoutedPrompt)
from prprompt.train import fit_predict, MAIN_METHODS, DIAGNOSTIC_METHODS, state_hash
from prprompt.protocol import (sample_support, support_folds, select_config,
                               seal_predictions, evaluate_sealed)


class ScientificContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        torch.set_num_threads(2)
        self.x = torch.randn(18, 7)
        src = torch.arange(18)
        self.edges = torch.stack([torch.cat([src, (src+1)%18]), torch.cat([(src+1)%18, src])])
        self.labels = torch.arange(18) % 3
        self.support = torch.arange(9)
        self.sy = self.labels[self.support]
        self.query = torch.arange(9, 18)
        self.backbone = TwoLayerGCN(7, 8, 8)
        self.config = {"steps": 3, "lr": .01, "weight_decay": .0001, "temperature": .1, "amplitude": 10., "gpf_tokens": 4}

    def test_all_methods_finite_and_freezing(self):
        original = state_hash(self.backbone)
        for method in MAIN_METHODS + DIAGNOSTIC_METHODS:
            with self.subTest(method=method):
                scores, state, audit = fit_predict(method, self.backbone, self.x, self.edges, self.support, self.sy, 3, self.config, 5)
                self.assertEqual(scores.shape, (18, 3))
                self.assertTrue(torch.isfinite(scores).all())
                self.assertEqual(audit["trainable_parameters"], sum(audit[k] for k in
                                 ["prompt_parameters", "head_parameters", "backbone_parameters", "task_model_parameters"]))
                self.assertEqual(state_hash(self.backbone), original)
                if audit["backbone_frozen"]:
                    self.assertEqual(audit["backbone_before"], audit["backbone_after"])
                elif method != "MLP":
                    self.assertNotEqual(audit["backbone_before"], audit["backbone_after"])
                if method == "PRPrompt":
                    self.assertGreater(audit["prompt_gradient_max"], 0)
                    self.assertGreater(float(state["prompt"]["tokens"].abs().sum()), 0)
                    self.assertEqual(audit["prompt_parameters"], 3*8)
                    self.assertLessEqual(audit["zero_prompt_max_error"], 1e-6)

    def test_label_renaming_equivariance_and_noop(self):
        adj = normalized_adjacency(self.edges, len(self.x), self.x.dtype)
        features = attribute_features(self.x, adj)
        q, probs = prototype_route(features, adj, self.support, self.sy, 3)
        renamed, _ = prototype_route(features, adj, self.support, (self.sy+1)%3, 3)
        self.assertTrue(torch.allclose(renamed[:, [1, 2, 0]], q, atol=1e-6))
        self.assertTrue(torch.allclose(q.sum(1), torch.zeros(18), atol=1e-6))
        hidden = self.backbone.conv_layers[0](self.x, self.edges)
        prompt = PrototypeRoutedPrompt(q, hidden)
        self.assertTrue(torch.equal(prompt(hidden), hidden))

    def test_stratified_sampling_and_disjoint_folds(self):
        s = sample_support(self.support, self.sy, 3, 2, 42)
        self.assertEqual(torch.bincount(self.labels[s]).tolist(), [2, 2, 2])
        self.assertFalse(torch.isin(s, self.query).any())
        self.assertTrue(torch.equal(s, sample_support(self.support, self.sy, 3, 2, 42)))
        for train, ty, valid, vy in support_folds(s, self.labels[s], 3):
            self.assertFalse(torch.isin(train, valid).any())
            self.assertEqual(len(ty.unique()), 3)
            self.assertEqual(len(vy.unique()), 3)

    def test_one_shot_no_extra_labels(self):
        config = {"fit": self.config, "selection": {"learning_rates": [.001, .01], "weight_decays": [0.]}}
        chosen, audit = select_config("PRPrompt", self.backbone, self.x, self.edges, torch.arange(3), self.labels[:3], 3, config, 1)
        self.assertEqual(chosen, self.config)
        self.assertEqual(audit["selection"], "one_shot_predeclared")

    def test_changed_support_changes_routing(self):
        adj = normalized_adjacency(self.edges, len(self.x), self.x.dtype)
        features = attribute_features(self.x, adj)
        q, _ = prototype_route(features, adj, self.support, self.sy, 3)
        changed = self.sy.clone()
        changed[0], changed[1] = self.sy[1], self.sy[0]
        r, _ = prototype_route(features, adj, self.support, changed, 3)
        self.assertFalse(torch.allclose(q, r))

    def test_query_labels_cannot_affect_fit_and_tamper_rejected(self):
        scores1, _, _ = fit_predict("PRPrompt", self.backbone, self.x, self.edges, self.support, self.sy, 3, self.config, 5)
        changed_labels = self.labels.clone()
        changed_labels[self.query] = (changed_labels[self.query]+1)%3
        scores2, _, _ = fit_predict("PRPrompt", self.backbone, self.x, self.edges, self.support,
                                    changed_labels[self.support], 3, self.config, 5)
        self.assertTrue(torch.equal(scores1, scores2))
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                evaluate_sealed(root, self.labels[self.query])
            seal_predictions(root, scores1, self.support, self.query, {"test": True})
            result = evaluate_sealed(root, self.labels[self.query])
            self.assertIn("accuracy", result)
            with (root / "predictions.npz").open("ab") as f:
                f.write(b"tampered")
            with self.assertRaises(ValueError):
                evaluate_sealed(root, self.labels[self.query])


if __name__ == "__main__":
    unittest.main(verbosity=2)
