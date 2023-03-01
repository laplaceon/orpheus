from setuptools import setup

setup(
    name='orpheus',
    version='0.1.1',
    description='A neural network for music representation learning',
    url='https://github.com/laplaceon/orpheus',
    author='Riyadh Rahman',
    packages=['orpheus.core', 'orpheus.model'],
    install_requires=[
        'torch',
        'numpy',
        'torchaudio',
        'x-transformers',
        'einops'
    ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
