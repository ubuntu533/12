from setuptools import setup

setup(
    name="jupyter-code-library",
    version="0.1.0",
    packages=["jupyter_code_library"],
    package_dir={"jupyter_code_library": "."},
    package_data={"jupyter_code_library": ["programs/*.py"]},
    include_package_data=True,
)
