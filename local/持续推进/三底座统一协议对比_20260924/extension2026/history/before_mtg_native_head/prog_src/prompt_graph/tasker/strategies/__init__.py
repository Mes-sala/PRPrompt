"""Prompt-type strategy implementations.

Each strategy encapsulates the train/eval logic for one ``prompt_type`` value
(``'None'``, ``'GPF'``, ``'GPPT'``, etc.). Importing this package registers
all bundled strategies into ``STRATEGY_REGISTRY``.
"""

from . import (
    input_structure,  # noqa: F401 -- SUPT and GraphTOP
    all_in_one,  # noqa: F401 -- import side-effect registers AllInOneStrategy
    dagprompt,  # noqa: F401 -- import side-effect registers DAGPrompTStrategy
    dhbp,  # noqa: F401 -- import side-effect registers DHBPStrategy
    edge_prompt,  # noqa: F401 -- import side-effect registers EdgePromptStrategy
    gpf,  # noqa: F401 -- import side-effect registers GPFStrategy + GPFPlusStrategy
    gppt,  # noqa: F401 -- import side-effect registers GPPTStrategy
    gprompt,  # noqa: F401 -- import side-effect registers GpromptStrategy
    graph_prompter,  # noqa: F401 -- import side-effect registers GraphPrompterStrategy
    multi_gprompt,  # noqa: F401 -- import side-effect registers MultiGpromptStrategy
    node_prototype,  # noqa: F401 -- frozen node baselines
    none,  # noqa: F401 -- import side-effect registers NoneStrategy
    pro_no_g,  # noqa: F401 -- import side-effect registers ProNoGStrategy
    prodigy,  # noqa: F401 -- import side-effect registers ProdigyStrategy
    psp,  # noqa: F401 -- import side-effect registers PSPStrategy
    relief,  # noqa: F401 -- import side-effect registers RELIEFStrategy
    self_pro,  # noqa: F401 -- import side-effect registers SelfProStrategy
    uni_prompt,  # noqa: F401 -- import side-effect registers UniPromptStrategy
)
