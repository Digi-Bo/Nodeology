from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [
        line.strip() for line in fh if line.strip() and not line.startswith("#")
    ]

setup(
    name="nodeology",
    version="0.0.2",
    author="Xiangyu Yin",
    author_email="xyin@anl.gov",
    description="Foundation AI-Enhanced Scientific Workflow",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/xyin-anl/nodeology",
    packages=find_packages(),
    classifiers=[
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
)
