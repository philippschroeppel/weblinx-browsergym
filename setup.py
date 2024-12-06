from setuptools import setup, find_packages

package_name = "weblinx_browsergym"
version = {}
with open(f"{package_name}/version.py") as fp:
    exec(fp.read(), version)

# with open("README.md") as fp:
#     long_description = fp.read()

long_description = """
# BrowserGym integration for the WebLINX benchmark

This package provides a BrowserGym environment for the WebLINX benchmark. It is built on top of the [BrowserGym](https://www.github.com/ServiceNow/browsergym) and [WebLINX](https://www.github.com/McGill-NLP/weblinx) packages.

## Installation

To install the package, run:
    
```bash
pip install weblinx-browsergym
```
"""

extras_require = {
    "dev": ["black", "wheel"],
}
 
# Dynamically create the 'all' extra by combining all other extras
extras_require["all"] = sum(extras_require.values(), [])

setup(
    name=package_name,
    version=version["__version__"],
    author="McGill NLP",
    author_email=f"weblinx@googlegroups.com",
    url=f"https://github.com/McGill-NLP/{package_name}",
    description=f"BrowserGym integration for the WebLINX benchmark",
    long_description=long_description,
    packages=find_packages(include=[f"{package_name}*"]),
    # package_data={package_name: ["_data/*.json"]},
    install_requires=[
        "tqdm",
        "huggingface_hub",
        "numpy",
        "weblinx[eval]>=0.3.2,<0.4.0",
        "datasets",
        "Pillow",
        "playwright",
        "browsergym-core>=0.11.2",
        "lxml",
        "gymnasium",
    ],
    extras_require=extras_require,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.10",
    # Cast long description to markdown
    long_description_content_type="text/markdown",
)