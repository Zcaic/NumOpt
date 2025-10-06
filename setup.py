"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
from pathlib import Path

# Get the long description from the README file
README_path = Path(__file__).parent / "README.md"
with open(README_path, encoding='utf-8') as f:
    long_description = f.read()

### Get the version number dynamically
init_path = Path(__file__).parent / "NumOpt" / "__init__.py"

with open(init_path) as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[1].strip().strip('"').strip("'")
            break
    else:
        raise RuntimeError("Unable to find version string.")

### Do the setup
setup(
    name='NumOpt',
    author='Zcaic',
    version=version,
    description='Opti is a Python package that helps you design and optimize engineered systems.',
    long_description=long_description,
    url='https://github.com/Zcaic/NumOpt.git',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    keywords='optimization automatic differentiation',
    packages=find_packages(),
    python_requires='>=3.9',
    install_requires=[
        'aerosandbox >= 4.2.8',
        "termcolor >= 3.1.0"
    ]
)