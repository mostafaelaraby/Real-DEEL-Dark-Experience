import sys
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
packages = setuptools.find_namespace_packages(include=["real_deel*"])
print("PACKAGES FOUND:", packages)
print(sys.version_info)

setuptools.setup(
    name="real_deel_dark_experience",
    version="0.0.1",
    author="Mostafa ElAraby, Yann Pequignot",
    author_email="<TODO>",
    description="Import REAL-DEEL entry to be used in Sequoia.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="TODO",
    packages=packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "Method": [
            "real_deel = real_deel_dark_experience.der:DER",
        ],
    },
    python_requires='>=3.7',
    install_requires=[
        "matplotlib==3.3.2",
        "torch_optimizer==0.1.0",
    ],
)