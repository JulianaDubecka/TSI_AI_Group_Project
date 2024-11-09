# TSI AI Group Project

A collaborative AI project developed by TSI Master's students, focusing on designing and implementing an AI-based solution to a real-world problem. In groups of three, students design and implement an AI-based solution to tackle a real-world problem. The project involves applying AI techniques like machine learning and data analysis in a functional model, comprehensive report, and final presentation. This repository contains the code, documentation, and resources for our group project.

## **Topic**

## **Team Members**
- **Member 1 Name** – Role / Area of Responsibility
- **Member 2 Name** – Role / Area of Responsibility
- **Member 3 Name** – Role / Area of Responsibility

## **Tools & Technologies**
- **Programming Language(s)**: Python, etc.
- **Libraries & Frameworks**: PyTorch, pandas, etc.
- **Development Tools**: Jupyter Notebook, etc.

## **Requirements**
- **Operating system:** This project was implemented and tested on MacOS Sequoia 15.1.
- **Python version:** Python 3.9.2 or above

## **Setting up the environment**
1. Clone the repository  
First, clone this repository to your local machine:
```bash
git clone https://github.com/JulianaDubecka/TSI_AI_Group_Project.git
cd TSI_AI_Group_Project
```
2. Install Conda (If Not Already Installed)  
If you don’t have Conda installed, download and install it from [Conda’s official website](https://anaconda.org/anaconda/conda).
3. Create the Conda Environment  
- **Using the environment.yml File**
   1. Create the environment using the `environment.yml` file:
    ```bash
    conda env create -f environment.yml -n
    ```
   2. Activate the environment
   ```bash
   conda activate sasrec_env
   ```
- **Using the `requirements.txt` File (Optional)**  
If you only have a requirements.txt file (for pip installation):
   1. Create a new Conda environment:
    ```bash
  conda create --name sasrec_env python=3.9.2
  ```
  2. Activate the environment:
  ```bash
  conda activate sasrec_env
  ```
  3. Install packages using `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```
4. Verifying the Installation  
To verify the environment setup, run the following command to check if all necessary packages are installed:
```bash
conda list
```

## **Troubleshooting**
- **Environment Conflicts**: If you encounter conflicts, try updating Conda and re-creating the environment.
```bash
conda update -n base -c defaults conda
conda env create -f environment.yml -n sasrec_env
```
- **Package Issues**: Some packages might not be available through Conda. In this case, manually install them with pip:
```bash
pip install <package-name>
```


## **Dataset**


## **Project Plan**
- **Milestones & Timeline**: []
  - **Phase 1**: []
  - **Phase 2**: []
  - **Phase 3**: []
  - **Phase 4**: []
  - **Phase 5**: []

## **Implementation**
[]

## **Results & Evaluation**
[]


