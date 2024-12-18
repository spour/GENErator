# GENErator

Torch implementation of a diffusion model designed for DNA sequence reconstruction and score prediction tasks, continuous, e.g. given a ChIP score, create a likely DNA sequence that might get it. 

**N.B. GRAHAM ON COMPUTE CANADA IS DOWN UNTIL JAN 7 -- THIS CODE IS NOT UP TO DATE AND MISSES A LOT OF STUFF THAT I CAN NOT ACCESS.**

## Features
- handle DNA sequences with **variable lengths** using padding and masking (up to a max)
- combined sequence reconstruction and score prediction objectives, helps to learn better
---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/dna-diffusion-model.git
   cd dna-diffusion-model
   pip install -e .
   ```
2. Data should look like. 
<score>    <sequence>
1.23       ATCGTAA
0.89       GCTAGCTGCTA

3. Modify main to look like.
   ```python 
   data_generator = BEDFileDataGenerator(
    filepath='/path/to/your/dataset.bed', 
    num_sequences=15000, 
    maxlen=512
   )
   ```

4. Run like:
```python
python main.py
```



