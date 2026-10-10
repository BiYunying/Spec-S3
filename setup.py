from setuptools import find_packages, setup
from pathlib import Path

setup(
    name="spec-s3",
    version="0.1.0",
    description="Spectrum-centric molecular representation and generation",
    packages=find_packages(),
    python_requires=">=3.11",
    install_requires=[
        line.strip()
        for line in (Path(__file__).parent / "requirements.txt")
        .read_text()
        .splitlines()
        if line.strip() and not line.startswith(("#", "--"))
    ],
)
