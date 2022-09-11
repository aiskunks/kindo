import glob
import typing
from pathlib import Path

save_path = "saved"


def set_save_path(path) -> None:
    global save_path
    save_path = path


def abs_path(local_path) -> str:
    return str(Path(local_path).resolve())


def get_saved_environments() -> typing.List[str]:
    return [env_path.replace("saved/", "") for env_path in glob.glob(f"{save_path}/*")]


def get_trained_model_paths(env_name) -> typing.List[str]:
    return [model_path for model_path in glob.glob(f"{save_path}/{env_name}/*")]


def get_trained_model_names(env_name) -> typing.List[str]:
    return [model_path.split("/")[-1] for model_path in glob.glob(f"{save_path}/{env_name}/*")]
