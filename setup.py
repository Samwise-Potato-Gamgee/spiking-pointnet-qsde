from setuptools import setup, find_packages

setup(
    name='spk_pointnet',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    python_requires='>=3.9',
    install_requires=[
        'torch>=2.2.0',
        'spikingjelly==0.0.0.0.14',
        'numpy>=1.26.0',
        'h5py>=3.10.0',
        'scipy>=1.12.0',
        'tqdm>=4.66.0',
        'pyyaml>=6.0',
        'matplotlib>=3.8.0',
        'seaborn>=0.13.0',
        'pandas>=2.2.0',
        'scikit-learn>=1.4.0',
    ],
    extras_require={
        'dev': ['jupyter', 'ipykernel', 'wandb'],
    },
    description='Spiking PointNet with Queue-Driven Sampling Direct Encoding (Q-SDE)',
    author='Samwise',
    license='MIT',
)
