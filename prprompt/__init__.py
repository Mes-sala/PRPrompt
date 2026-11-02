"""Prototype-routed input prompts for few-shot node and graph classification."""
from .backbone import TwoLayerGCN
from .graph import GraphReference, GraphFit, GraphInputPrompt, prepare_graph_reference, fit_graph
from .node import fit_fullgraph, fit_subgraph, SubgraphConfig

__version__ = '3.0.0'
__all__ = ['TwoLayerGCN', 'GraphReference', 'GraphFit', 'GraphInputPrompt',
           'prepare_graph_reference', 'fit_graph', 'fit_fullgraph', 'fit_subgraph', 'SubgraphConfig']
