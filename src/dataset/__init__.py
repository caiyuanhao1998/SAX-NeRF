from .tigre import TIGREDataset
from .tigre_mlg import TIGREDataset_MLG

__all__ = ["TIGREDataset", "TIGREDataset_MLG", "TIGREDataset_Traditional"]


# TIGRE is only required by traditional reconstruction methods.
def __getattr__(name):
    if name == "TIGREDataset_Traditional":
        from .tigre_traditional import TIGREDataset_Traditional
        globals()[name] = TIGREDataset_Traditional
        return TIGREDataset_Traditional
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
