from setuptools import setup, find_packages

setup(
    name="jupyter-code-library",
    version="0.1.1",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "jupyter_code_library": ["programs/*.py"]
    }
)
