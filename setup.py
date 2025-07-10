#!/usr/bin/env python
from setuptools import setup, find_packages
import os
import re

# Read the version from __init__.py
with open(os.path.join('splendor_ai', '__init__.py'), 'r') as f:
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if version_match:
        version = version_match.group(1)
    else:
        version = '0.1.0'  # Default if not found

# Read long description from README.md
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Core dependencies
install_requires = [
    'torch>=2.0.0,<3.0.0',  # PyTorch CPU version
    'numpy>=1.22.0,<2.0.0',  # Numerical operations
    'matplotlib>=3.5.0,<4.0.0',  # Visualization
    'pandas>=1.4.0,<2.0.0',  # Data analysis
    'tensorboard>=2.10.0,<3.0.0',  # Training metrics visualization
    'pydantic>=2.0.0,<3.0.0',  # Data validation and settings management
    'rich>=12.0.0,<13.0.0',  # Beautiful terminal output
    'tqdm>=4.64.0,<5.0.0',  # Progress bars
]

# Development dependencies
dev_requires = [
    'pytest>=7.0.0,<8.0.0',  # Testing framework
    'pytest-cov>=4.0.0,<5.0.0',  # Test coverage
    'mypy>=1.0.0,<2.0.0',  # Static type checking
    'black>=23.0.0,<24.0.0',  # Code formatting
    'isort>=5.10.0,<6.0.0',  # Import sorting
]

setup(
    name='splendor-ai',
    version=version,
    description='A reinforcement learning and MCTS framework for the board game Splendor',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Splendor AI Team',
    author_email='your-email@example.com',  # Replace with your email
    url='https://github.com/yourusername/splendor-ai',  # Replace with your repository URL
    packages=find_packages(),
    install_requires=install_requires,
    extras_require={
        'dev': dev_requires,
        'all': dev_requires,
    },
    entry_points={
        'console_scripts': [
            'splendor-play=splendor_ai.play:main',
            'splendor-train=splendor_ai.train:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Games/Entertainment :: Board Games',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.9',
    include_package_data=True,
    keywords='splendor, board game, ai, reinforcement learning, mcts, monte carlo tree search',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/splendor-ai/issues',
        'Source': 'https://github.com/yourusername/splendor-ai',
    },
)
