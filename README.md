# template-Parkinson-Risk-Assessment

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![NumPy](https://img.shields.io/badge/NumPy-1.21%2B-013243?style=for-the-badge&logo=numpy)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-%23ff69b4?style=for-the-badge)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas)
![Seaborn](https://img.shields.io/badge/Seaborn-Data%20Visualization-3776AB?style=for-the-badge)

## Project Overview

This project's end goal is to assess Parkinson's disease risk using a simple machine learning mode, training on vocal biomarkers. We will analyze the dataset (`parkinsons.data`) to identify and train patterns that distinguish characteristics in individuals with Parkinson's disease from healthy individuals. 

## What is Parkinsons disease?

Parkinson's disease was described by James Parkinson in 1817 as a "shaking palsy". Parkinson's disease is a neurodegenerative disorder of the brain that results in a loss of dopamine-producing neurons in the region called substantia nigra. Loss of this neurotransmitter dopamine causes neurons to fire randomly, leading to the symptoms of Parkinson's disease (listed below). Moreover, people with Parkinson's disease also lose norepinephrine, a chemical messenger that controls many of the body's functions.

The exact cause of Parkinson's disease is not fully understood, however, there exist several factors that seem to influence the risk, including:
- <b>Specific Genes</b>: Specific genetic changes are linked to Parkinson's disease, however, these are rare unless many family members have had Parkinson's disease.
- <b>Environmental Factors</b>: Exposure to toxins and other environmental factors may increase the risk of Parkinson's disease. For example, <a href="https://en.wikipedia.org/wiki/MPTP">MPTP</a> a substance found in many illegal drugs is a neurotoxin that can cause Parkinson's disease-like symptoms. Other examples include pesticides and well water for drinking. It is important to note that no environmental factor has proven to be the cause.

Research on Parkinson's disease has shown that the brain undergoes significant changes. These changes include:
- <b>Lewy bodies</b>: Clumps of proteins in the brain, namely Lewy bodies are associated with Parkinson's disease. Researchers believe these proteins hold an important clue to the cause.
- <b>Alpha-synuclein within the Lewy bodies</b>: Alpha-synuclein is a protein found in all Lewy bodies. Moreover, interestingly, Alpha-synuclein proteins have been found in the spinal fluid of people who later got Parkinson's disease.
- <b>Altered mitochondria</b>: Mitochondria are powerhouse factories inside cells that create a significant portion of the body's energy.  Changes in the mitochondria have been found in the brains of those with Parkinson's disease.

## Symptoms

Parkinson's symptoms may include the following:
- <b>Tremor</b>: A tremble shaking usually begins in the hands or fingers. Additionally, the tremor can begin in the foot or jaw.
- <b>Slowed movement (Bradykinesia)</b>: Parkinson's disease can slow your movement, making simple tasks significantly more difficult.
- <b>Rigid Muscles</b>: Muscles may feel tense and painful, and arm movements may be short and jerky.
- <b>Poor Posture and Balance</b>: Posture may begin to sink, and may have balance problems.
- <b>Loss of automatic movements</b>: Difficulty performing movements that are traditionally done automatically, such as blinking, smiling, etc...
- <b>Writing Changes</b>: Difficulty writing, and the writing may appear cramped and small.
- <b>Nonmotor symptoms</b>: These symptoms could include depression, anxiety, sleep problems, difficulty smelling, and problems with thinking and memory.
- <b>Voice changes</b>: Parkinson's disease can affect one voice, including a quieter voice, hoarseness, and slurred speech (What we will be looking at).

## Resources

<a href="https://www.mayoclinic.org/diseases-conditions/parkinsons-disease/symptoms-causes/syc-20376055">Mayo Clinic</a>, <a href="https://med.stanford.edu/parkinsons/introduction-PD/history.html#:~:text=First%20described%20in%201817%20by,of%20cells%20that%20produce%20dopamine.">Stanford Medicine</a>

## Starting the project

### Understanding the Data

For this project, you will use the `parkinsons.data` dataset (Max A. Little, Patrick E. McSharry, Eric J. Hunter, Lorraine O. Ramig). It is important to understand the variables and the dataset.

#### Matrix column entries (attributes):
- name - ASCII subject name and recording number
- MDVP:Fo(Hz) - Average vocal fundamental frequency
- MDVP:Fhi(Hz) - Maximum vocal fundamental frequency
- MDVP:Flo(Hz) - Minimum vocal fundamental frequency
- MDVP:Jitter(%), MDVP:Jitter(Abs), MDVP:RAP, MDVP:PPQ, Jitter:DDP - Several measures of variation in fundamental frequency
- MDVP:Shimmer,MDVP:Shimmer(dB),Shimmer:APQ3,Shimmer:APQ5,MDVP:APQ,Shimmer:DDA - Several measures of variation in amplitude
- NHR, HNR - Two measures of the ratio of noise to tonal components in the voice
- status - The health status of the subject (one) - Parkinson's, (zero) - healthy
- RPDE, D2 - Two nonlinear dynamical complexity measures
- DFA - Signal fractal scaling exponent
- spread1,spread2,PPE - Three nonlinear measures of fundamental frequency variation

### Libraries / Dependencies

For this project, we will use the `pandas`, `numpy`, `matplotlib`, and `seaborn` libraries. If you haven't already, please install these libraries using pip/pip3 install.

### Exploratory Data Analysis (EDA)

The first look at your data. During this EDA process, you are trying to understand the data's characteristics, identify patterns, and uncover potential insights by examining its structure, relationships, and anomalies.

1. The first step, will be to import all the libraries previously mentioned.
2. We want to read the dataset using the `read_csv` function from the `pandas` library. The dataset variable is generally `df`, which is what you will be using to call the functions below.
3. Next we want to use the following functions for EDA:
     * `info()` provides a quick summary of the info about the data frame.
     * `describe()` provides high-level summaries of all the columns in the dataset. Including count, mean, std, min, max, and 25%, 50%, and 75% percentiles of the values.
     * `hist()` creates histograms for each column of the dataset. You can play around with the figure size, by passing `figsize=(x,y)` as a parameter.
     * `isnull().sum()` provides a summary of the number of "missing values" for each of the columns. However, there is a caveat, you must look at the columns, and verify that a null value is NOT logical, thus it IS a missing value. There are multiple ways to address null values, which will be explored later.
     * `corr()` provides a data frame summary of the computed pairwise correlation of columns in the dataset. However, this can be intimidating and also hard to understand. Thus we can also visualize in a heatmap using the `seaborn` library.
4. Using the seaborn library create a heatmap showing the pairwise correlation of all the columns in the dataset. **Hint:** This should be 3-4 lines of code.
5. Based off all of these functions, what sort of information and conclusions can you draw from the EDA?

> Insert your graphs and summaries here & comment out the above instructions.

### Support Vector Machine (SVM) 



