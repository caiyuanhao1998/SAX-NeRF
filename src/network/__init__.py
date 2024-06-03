from .network import DensityNetwork
from .Lineformer import Lineformer


def get_network(type):
    if type == "mlp":
        return DensityNetwork
    elif type == "Lineformer":
        return Lineformer
    else:
        raise NotImplementedError("Unknown network type!")

