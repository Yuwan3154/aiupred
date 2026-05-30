# AIUPred v1.2.1

AIUPred is free to use for academic users. It contains a loadable python library as well as an executable python script. First download and extract AIUPred. Change the working directory to the extracted directory and install its dependencies. It is highly advised to use a virtual environment!

`pip3 install -r requirements.txt`

If you want to use the standalone executable just run

`python3 aiupred.py`

Available options:

```usage: aiupred.py [-h] -i INPUT_FILE [-o OUTPUT_FILE] [-v] [-g GPU] [--force-cpu] [--no-smoothing] [--low-memory [LOW_MEMORY]]

options:
  -h, --help            show this help message and exit
  -i INPUT_FILE, --input_file INPUT_FILE
                        Input file in (multi) FASTA format
  -o OUTPUT_FILE, --output_file OUTPUT_FILE
                        Output file
  -v, --verbose         Increase output verbosity
  -g GPU, --gpu GPU     Index of GPU to use, default=0
  --force-cpu           Force the network to only utilize the CPU. Calculation will be very slow, not recommended
  --no-smoothing        Removes the default SavGol smoothing function
  --low-memory [LOW_MEMORY]
                        Use chunking to lower the memory usage. Default chunk size is 1000. The lower the chunk size the lower the memory consumption well as the accuracy
```

The following section gives some tips how to use the importable library.

Add the location of the extracted directory to your PYTHONPATH environment variable (assuming standard bash shell)

`export PYTHONPATH="${PYTHONPATH}:/path/to/aiupred/folder"`

After reloading the shell AIUPred will be importable in your python scripts.

```import aiupred_lib
# Load the models and let AIUPred find if a GPU is available.     
embedding_model, regression_model, device = aiupred_lib.init_models()
# Predict disorder of a sequence
sequence = 'THISISATESTSEQENCE'
prediction = aiupred_lib.predict_disorder(sequence, embedding_model, regression_model, device)
```

# Primary citation:
```
AIUPred: combining energy estimation with deep learning for the enhanced prediction of protein disorder
Gábor Erdős, Zsuzsanna Dosztányi
Nucleic Acids Research 2024; gkae385 
```