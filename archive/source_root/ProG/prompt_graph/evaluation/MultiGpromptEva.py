import torch

from prompt_graph.utils import get_logger

from .node_scores import classification_scores

logger = get_logger(__name__)


def MultiGpromptEva(
    test_embs, test_lbls, idx_test, prompt_feature, Preprompt, DownPrompt, sp_adj, num_class, device
):
    Preprompt.eval()
    with torch.no_grad():
        embeds1, _ = Preprompt.embed(prompt_feature, sp_adj, True, None, False)
        test_embs1 = embeds1[0, idx_test]
        logger.debug("idx_test %s", idx_test)
        # ``test_lbls`` is passed only as the metric target.  DownPrompt's
        # default train=0 means it does not update its support prototypes.
        logits = DownPrompt(test_embs, test_embs1, test_lbls, train=0)
    return classification_scores(logits, test_lbls)
