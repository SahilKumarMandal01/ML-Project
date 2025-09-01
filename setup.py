from setuptools import find_packages, setup
from typing import List

HYPERN_E_DOT = "-e ."
def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements
    """
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        requirements = [req.replace("\n", "") for req in requirements]

        if HYPERN_E_DOT in requirements:
            requirements.remove(HYPERN_E_DOT)
    
    return requirements

setup(
    name="mlproject",
    version="0.0.1",
    author="Sahil Kumar Mandal",
    author_email="thesahilmandal@outlook.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt")
)