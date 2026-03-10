from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="fireworks",
    version="0.1.0",
    author="DanielTParry",
    author_email="dan.t.parry@gmail.com",
    description="A library for financial independence and retirement planning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DanielTParry/FIREworks",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
