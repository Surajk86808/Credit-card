from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()
    
setup(
    name = "CreditCardFraudDetection",
    version = "0.0.1",
    author = "Suraj Kumar",
    packages = find_packages(),
    install_requires = requirements,
    description = "A previous data base fruad detection project",
)         