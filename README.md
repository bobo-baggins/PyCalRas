# PyCalRas

## Setup Instructions

### 1. Install Conda

If you don't have Conda installed, follow these steps:

1. Download the Miniconda installer for your operating system from [here](https://docs.conda.io/en/latest/miniconda.html).
2. Run the installer and follow the prompts.
3. After installation, open a new terminal window to ensure Conda is initialized.

To verify the installation, run:

```
conda --version
```

### 2. Create and Activate the Conda Environment

1. Navigate to the project directory containing the `environment.yml` file.

2. Create the Conda environment using the following command:

```
conda env create -f environment.yml
```

3. Activate the newly created environment:

```
conda activate PyCal
```

### 3. Install Additional Dependencies

If you need to install additional dependencies, you can do so using the following command:

```
conda install <package_name>
```

Replace `<package_name>` with the name of the package you want to install.

### 4. Setting Up in PyCharm

1. Open PyCharm and go to **File > Settings** (or **PyCharm > Preferences** on macOS).
2. In the left pane, select **Project: [Your Project Name] > Python Interpreter**.
3. Click the gear icon ⚙️ and select **Add**.
4. Choose **Conda Environment** and select **Existing environment**.
5. Browse to the location of your Conda environment (usually found in `~/miniconda3/envs/PyCal` or `~/anaconda3/envs/PyCal`).
6. Select the `python.exe` file in the `bin` directory (or `Scripts` directory on Windows).
7. Click **OK** to apply the changes.

### 5. Setting Up in Spyder

1. Open Spyder.
2. Go to **Tools > Preferences**.
3. In the left pane, select **Python Interpreter**.
4. Choose **Use the following Python interpreter**.
5. Browse to the location of your Conda environment (usually found in `~/miniconda3/envs/PyCal` or `~/anaconda3/envs/PyCal`).
6. Select the `python.exe` file in the `bin` directory (or `Scripts` directory on Windows).
7. Click **OK** to apply the changes.

### 6. Running the Project

Now you can run your project in either PyCharm or Spyder using the Conda environment you just set up.
