import setuptools

setuptools.setup(
    name="semlog",
    version="0.3.0",
    author="Christian Dormagen",
    author_email="christian.dormagen@oth-regensburg.de",
    description="OCED based OWL ontology construction and instantiation for event log data",
    packages=["semlog"],
    include_package_data=True,
    install_requires=[
    ],
    entry_points={
        'console_scripts': [
            'semlog = semanticlog.run.cli:main',
        ],
    },
)