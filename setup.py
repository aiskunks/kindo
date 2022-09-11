from setuptools import find_packages, setup


def read_reqs(req_file: str):
    with open(req_file) as req:
        return [
            line.strip()
            for line in req.readlines()
            if line.strip() and not line.strip().startswith("#")
        ]


setup(
    name="kindo",
    packages=find_packages(),
    author="Fedor Grab",
    author_email="grab.f@northeastern.edu",
    url="https://github.com/NEU-AI-Skunkworks/kindo",
    description=(
        "High level API for Reinforcement Learning frameworks TF-Agents and Stable-Baselines"
    ),
    license="MIT",
    install_requires=read_reqs("requirements.txt"),
    include_package_data=True,
    version="1.0.0",
    zip_safe=False,
)
