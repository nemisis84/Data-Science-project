# Data-Science-project

## Setup environment

Start by installing Python 3.12 from the [official website](https://www.python.org/downloads/). Alternatively, Unix users can download it by running the following command:

```bash
sudo apt install python3.12
```

However, Unix users will then need to run all Python commands using python3.12.

### Windows

1. Open a terminal (powershell or cmd) and navigate to the project directory.
2. Ensure the correct python version is in PATH, by running

```bash
python --version
```

```bash
python -m venv .venv
```

4. Activate the virtual environment by running one of the following commands, depending on your terminal:

- Windows cmd: `.venv\Scripts\activate.bat`
- Powershell: `.venv\Scripts\activate.ps1`

5. Install the requirements by running

```bash
python -m pip install -r requirements.txt
```

### Unix (macOS)

1. Open a terminal and navigate to the project directory.
2. Initialize the virtual environment by running:

```bash
python3 -m venv .venv
```

3. Activate the virtual environment by running:

```bash
source .venv/bin/activate
```

4. Install the requirements by running:

```bash
python3 -m pip install -r requirements.txt
```

## Running code 

To run the code in terminal, change to the labs folder and run `python <script_name>.py`. Alternatively, you can run the code in your IDE of choice by setting up the venv as an interpreter.

### Deactivate and delete the virtual environment

To deactive the virtual environment, simply run the command `deactivate`. If you want to reactivate an existing virtual environment, follow the same instructions about activating a virtual environment. There’s no need to create a new virtual environment.

To delete the virtual environment, delete the folder ".venv" in the project directory.