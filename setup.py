from setuptools import setup

setup_kwargs = {
    'name': 'pypermut',
    'version': '0.2.2',
    'description': 'Python package for permutation tests, for statistics and machine learning',
    'long_description': 'Python package for permutation tests, for statistics and machine learning',
    'author': 'Quentin Barthelemy',
    'url': 'https://github.com/qbarthelemy/PyPermut',
    'packages': ['pypermut'],
    'platforms': 'any',
    'python_requires': '>=3.9,<4.0',
    'install_requires': [
        'matplotlib>=3.0.0',
        'numpy>=1.16.5',
        'scipy>=1.6.0,<1.17.0',
    ],
    'extras_require': {
        'docs': [
            'sphinx',
            'sphinx-gallery',
            'numpydoc',
            'scikit-learn',
        ],
        'tests': [
            'flake8',
            'pytest',
        ],
    },
}


setup(**setup_kwargs)
