from .stable_audio import StableAudioGenerator
from .musicgen import MusicGenGenerator

from .musicgen_melody import MusicGenGenerator as MusicGenMelodyGenerator
from .musiccontrollite import MusicControlLiteGenerator

__all_unc__ = ["StableAudioGenerator", "MusicGenGenerator"]
__all_con__ = ["MusicControlLiteGenerator", "MusicGenMelodyGenerator"]
# __all__ = ["MusicGenGenerator"]
__all__ = __all_unc__ + __all_con__
