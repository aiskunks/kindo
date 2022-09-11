import secrets
import typing

from stable_baselines3.common.base_class import BaseAlgorithm


def chunks(lst: typing.List, size: int):
    """Yield successive n-sized chunks from list"""
    for i in range(0, len(lst), size):
        yield lst[i : i + size]


def compile_random_model_name(model: BaseAlgorithm) -> str:
    return f"{model.__class__.__name__}_{secrets.token_hex(2)}"
