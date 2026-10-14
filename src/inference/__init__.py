from .backend import InferenceBackend, SimulatedBackend
from .data_types import (
    DetectionResult,
    PipelineResult,
    ReIDResult,
    TrackResult,
    # Pipeline B
    PoseResult,
    BehaviorResult,
    PipelineBResult,
    # Pipeline C
    CropResult,
    LPRResult,
    PipelineCResult,
    # Pipeline D
    SegmentationResult,
    CrowdCountResult,
    PipelineDResult,
)

__all__ = [
    "InferenceBackend",
    "SimulatedBackend",
    "DetectionResult",
    "TrackResult",
    "ReIDResult",
    "PipelineResult",
    "PoseResult",
    "BehaviorResult",
    "PipelineBResult",
    "CropResult",
    "LPRResult",
    "PipelineCResult",
    "SegmentationResult",
    "CrowdCountResult",
    "PipelineDResult",
]

# Pipeline runners 需要额外依赖，延迟导入
try:
    from .pipeline_runner import PipelineRunner

    __all__.append("PipelineRunner")
except ImportError:
    pass

try:
    from .pipeline_runner_b import PipelineRunnerB

    __all__.append("PipelineRunnerB")
except ImportError:
    pass

try:
    from .pipeline_runner_c import PipelineRunnerC

    __all__.append("PipelineRunnerC")
except ImportError:
    pass

try:
    from .pipeline_runner_d import PipelineRunnerD

    __all__.append("PipelineRunnerD")
except ImportError:
    pass
