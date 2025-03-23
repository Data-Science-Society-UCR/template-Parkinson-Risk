# template-Parkinson-Risk-Assessment
```diff
# Python
+ testv2

```
#### Tags: Python, NumPy, Pandas, Matplotlib

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
- <b>Voice changes</b>: Parkinson's disease can affect one voice, including a quieter voice, hoarseness, and slurred speech.

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

For this project, we will use the `pandas`, `numpy`, and `matplotlib` (potentially `seaborn`) libraries. If you haven't already, please install these libraries using pip/pip3 install.



