
from pathlib import Path
from IPython import get_ipython

LIBRARY_PATH = Path(__file__).parent / "programs"

def show(program):
    file_path = LIBRARY_PATH / f"{program}.py"

    if not file_path.exists():
        print("Program not found!")
        return

    code = file_path.read_text(encoding="utf-8")
    get_ipython().set_next_input(code, replace=True)


def add(program, code):
    LIBRARY_PATH.mkdir(parents=True, exist_ok=True)

    file_path = LIBRARY_PATH / f"{program}.py"

    file_path.write_text(code, encoding="utf-8")

    print(f"{program} saved successfully!")
