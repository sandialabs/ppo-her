import setuptools

with open('README.md', 'r', encoding='utf-8') as file:
    long_description = file.read()

setuptools.setup(
    name="ppo_her",
    version="0.0.0",
    author="Cale Crowder",
    author_email="dccrowd@sandia.gov",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_dirs={"": "meher"},
    py_modules=[],
)
