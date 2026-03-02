from setuptools import setup, find_packages

setup(
    packages=find_packages(),
    package_data={
        'reg_normalizer': ['data/interim/*.yaml', 'data/interim/*.csv'],
    },
    include_package_data=True,
)
