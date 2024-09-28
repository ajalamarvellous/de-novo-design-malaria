# Hit Discovery for novel antimalarial Using AI

This project seeks to use AI to discover novel drugs for anti-malarial.

### Steps/Process
- [x] `Get all anti-malaria dataset from CHEMBL.`
- [ ] Use LoHi splitter to train model and identifiy antimalaria`
- [ ] `Try multiple models to predict whether antimalaria (gradient boosted,random forest, ANN, logistic regression, GCNN, GAT, Chemprop, MolChem)`
- [ ] Train CLM to predict new molecules (LSTM, GRU, RNN, Transformers, Mamba, VAE) on a large CHEMBL dataset
- [ ] `Finetune on subselected positive malaria datasets.`
- [ ] `Test the molecule to identify which are valid smiles conbination`
- [ ] `Test the molecule to identify which are positive.`
- [ ] `Dock and validate the top positive molecules.`
- [ ] Synthesis and perform wetlab evaluation of the positive.

<!-- FOLDER-STRUCTURE-->
## Folder structure
------------
```text
    ├── LICENSE            <- The Project Licence for the project.
    ├── CONDUCT.md         <- Code of Conduct for contributions.
    ├── README.md          <- The top-level README about the project.
    ├── requirements.txt   <- The requirements file for reproducing the project 
    │                        
    ├── src                <- Python scripts folder for the project.
    │    
    ├── Notebooks          <- Jupyter notebooks folder for the project 
    |         
    ├── models             <- Model artifacts folder for the project 
    │        
    ├── data               <- Data folder for the project from
    │   └── MalariaData_bioactivity   <- the project's primary dataset
    |     
    └── test    <- test suite for the project
         
```

<!-- BRANCHES-STRUCTURE-->
## Git Branches structure
------------
```text
    ├── main                  <- Main project backbone where everything is synced together.
    ├── virtual-screening     <- Machine learning screening section towards identifying great candidates from not so great ones.
    └── de-novo-design        <- Generative AI section towards de novo design of new candidates.

```

<!-- GETTING STARTED -->
## Getting Started

To get started and set up the project in your local environment, please download the packages listed in the requirements

### Installation

1. Create a virtual environment
    ```sh
    python -m venv .venv
    ```
2. Activate the virtual environment
    ```sh
    source .venv/bin/activate
    ```
3. Clone the repo
   ```sh 
   git clone git@github.com:ajalamarvellous/de-novo-design-malaria.git
   ```
4. Install the necessary packages
   ```sh
   pip install requirements
   ```
5. Checkout to the specific branch associated with the section you're working on
   
   for `virtual screening`
   ```sh
   git checkout virtual-screening
   ```
   
    for `de novo design`
   ```sh
   git checkout de-novo-design
   ```

   `Please, avoid directly coding into the main branch`
   And you are ready to rumble


<!-- DATA DOWNLOAD -->
## Data Download

The dataset used in this project is gotten from CHEMBL
To download the data used in this project in Google Colab or Kaggle space, create a new code block

```python
!wget https://github.com/ajalamarvellous/de-novo-design-malaria/raw/main/data/MalariaData_bioactivity.txt
```

And to read the file 
```python
import pandas as pd

df = pd.read_table("MalariaData_bioactivity.txt")
```

Voila, there you go, you should have the data running

<!-- SYNC-->
## Updating contribution

To ensure collaborations sync properly and avoid conflict, please ensure that you pull from the origin before you start coding

```sh
git pull origin <branch-name>
```

And once you're done coding your part before you log, please push it to the origin
```sh
git push origin <branch-name>
```

And there you go.... we should avoid conflicts this ways


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

Please check the `CONTRIBUTING.md` on important information to contribute to this project


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.



<!-- CONTRIBUTING -->
## Contributing

- [Ajala, Marvellous](http://ajalamarvellous.bio) 

- [Damilola S. Bodun](https://researchgate.net/profile/Bodun-Damilola)

- [Ayangoke Onilude](https://x.com/_ayangoke)

Project Link: [https://github.com/ajalamarvellous/de-novo-design-malaria](https://github.com/ajalamarvellous/de-novo-design-malaria)

