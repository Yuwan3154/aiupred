import logging
from scipy.signal import savgol_filter
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.functional import pad
import math
import os
import numpy as np
import multiprocessing as mp
from functools import partial
import pandas as pd
from glob import glob
from tqdm import tqdm
from Bio.PDB import PDBParser


PATH = os.path.dirname(os.path.realpath(__file__))
AA_CODE = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', 'X']
WINDOW = 100


class PositionalEncoding(nn.Module):
    """
    Positional encoding for the Transformer network
    """
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class TransformerModel(nn.Module):
    """
    Transformer model to estimate positional contact potential from an amino acid sequence
    """
    def __init__(self):
        super().__init__()
        self.d_model = 32
        self.pos_encoder = PositionalEncoding(self.d_model)
        encoder_layers = TransformerEncoderLayer(self.d_model, 2, 256, 0)
        self.transformer_encoder = TransformerEncoder(encoder_layers, 2)
        self.encoder = nn.Embedding(21, self.d_model)
        self.decoder = nn.Linear((WINDOW + 1) * self.d_model, 1)

    def forward(self, src: Tensor, embed_only=False) -> Tensor:
        src = self.encoder(src) * math.sqrt(self.d_model)
        src = self.pos_encoder(src)  # (Batch x Window+1 x Embed_dim)
        embedding = self.transformer_encoder(src)
        if embed_only:
            return embedding
        output = torch.flatten(embedding, 1)
        output = self.decoder(output)
        return torch.squeeze(output)


class DecoderModel(nn.Module):
    """
    Regression model to estimate disorder propensity from and energy tensor
    """

    def __init__(self):
        super().__init__()
        input_dim = WINDOW + 1
        output_dim = 1
        current_dim = input_dim
        layer_architecture = [16, 8, 4]
        self.layers = nn.ModuleList()
        for hdim in layer_architecture:
            self.layers.append(nn.Linear(current_dim, hdim))
            current_dim = hdim
        self.layers.append(nn.Linear(current_dim, output_dim))

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers[:-1]:
            x = torch.relu(layer(x))
        output = torch.sigmoid(self.layers[-1](x))
        return torch.squeeze(output)


@torch.no_grad()
def tokenize(sequence, device):
    """
    Tokenize an amino acid sequence. Non-standard amino acids are treated as X
    :param sequence: Amino acid sequence in string
    :param device: Device to run on. CUDA{x} or CPU
    :return: Tokenized tensors
    """
    return torch.tensor([AA_CODE.index(aa) if aa in AA_CODE else 20 for aa in sequence], device=device)


def predict_disorder(sequence, energy_model, regression_model, device, no_smoothing=False):
    """
    Predict disorder propensity from a sequence using a transformer and a regression model
    :param sequence: Amino acid sequence in string
    :param energy_model: Transformer model
    :param regression_model: regression model
    :param device: Device to run on. CUDA{x} or CPU
    :param smoothing: Use the SavGol filter to smooth the output
    :return:
    """
    predicted_energies = calculate_energy(sequence, energy_model, device)
    padded_energies = pad(predicted_energies, (WINDOW // 2, WINDOW // 2), 'constant', 0)
    unfolded_energies = padded_energies.unfold(0, WINDOW + 1, 1)
    predicted_disorder = regression_model(unfolded_energies).detach().cpu().numpy()
    if not no_smoothing and len(sequence) >= 10:
        predicted_disorder = savgol_filter(predicted_disorder, 11, 5)
    return predicted_disorder


def calculate_energy(sequence, energy_model, device):
    """
    Calculates residue energy from a sequence using a transformer network
    :param sequence: Amino acid sequence in string
    :param energy_model: Transformer model
    :param device: Device to run on. CUDA{x} or CPU
    :return: Tensor of energy values
    """
    tokenized_sequence = tokenize(sequence, device)
    padded_token = pad(tokenized_sequence, (WINDOW // 2, WINDOW // 2), 'constant', 20)
    unfolded_tokens = padded_token.unfold(0, WINDOW + 1, 1)
    return energy_model(unfolded_tokens)


def multifasta_reader(file_handler):
    """
    (multi) FASTA reader function
    :return: Dictionary with header -> sequence mapping from the file
    """
    sequence_dct = {}
    header = None
    for line in file_handler:
        if line.startswith('>'):
            header = line.strip()
            sequence_dct[header] = ''
        elif line.strip():
            sequence_dct[header] += line.strip()
    file_handler.close()
    return sequence_dct


def init_models(force_cpu=False, gpu_num=0):
    """
    Initialize networks and device to run on
    :param force_cpu: Force the method to run on CPU only mode
    :param gpu_num: Index of the GPU to use, default=0
    :return: Tuple of (embedding_model, regression_model, device)
    """
    device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
    if force_cpu:
        device = 'cpu'
    logging.debug(f'Running on {device}')
    if device == 'cpu':
        print('# Warning: No GPU found, running on CPU. It is advised to run AIUPred on a GPU')

    embedding_model = TransformerModel()
    embedding_model.load_state_dict(torch.load(f'{PATH}/data/embedding.pt', map_location=device))
    embedding_model.to(device)
    embedding_model.eval()

    reg_model = DecoderModel()
    reg_model.load_state_dict(torch.load(f'{PATH}/data/decoder.pt', map_location=device))
    reg_model.to(device)
    reg_model.eval()

    logging.debug("Networks initialized")

    return embedding_model, reg_model, device


@torch.no_grad()
def batch_predict_disorder(
    sequences,
    energy_model,
    regression_model,
    device,
    no_smoothing=False,
    window_batch_size=4096,
):
    """
    Predict disorder propensity for a list of sequences in one GPU pass.
    More efficient than calling predict_disorder() per sequence because all
    windows from all sequences are concatenated and processed together.

    :param sequences: List of amino acid sequence strings
    :param energy_model: TransformerModel (on device, eval mode)
    :param regression_model: DecoderModel (on device, eval mode)
    :param device: torch device
    :param no_smoothing: If True, skip SavGol smoothing
    :param window_batch_size: Max number of 101-token windows per GPU mini-batch
    :return: List of numpy arrays, one per sequence
    """
    if not sequences:
        return []
    lengths = [len(s) for s in sequences]

    # Phase 1: tokenize all sequences and build (L_i, WINDOW+1) token windows
    all_token_windows = []
    for seq in sequences:
        tokens = tokenize(seq, device)                                         # (L,)
        padded = pad(tokens, (WINDOW // 2, WINDOW // 2), 'constant', 20)      # (L+WINDOW,)
        all_token_windows.append(padded.unfold(0, WINDOW + 1, 1))             # (L, WINDOW+1)
    flat_tokens = torch.cat(all_token_windows, dim=0)                          # (sum_L, WINDOW+1)

    # Phase 2: energy model in mini-batches
    energy_chunks = []
    for i in range(0, flat_tokens.shape[0], window_batch_size):
        energy_chunks.append(energy_model(flat_tokens[i:i + window_batch_size]))
    flat_energies = torch.cat(energy_chunks)                                   # (sum_L,)

    # Phase 3: split energies back by sequence, build (L_i, WINDOW+1) energy windows
    all_energy_windows = []
    for energies in torch.split(flat_energies, lengths):
        padded_e = pad(energies, (WINDOW // 2, WINDOW // 2), 'constant', 0.0)
        all_energy_windows.append(padded_e.unfold(0, WINDOW + 1, 1))          # (L, WINDOW+1)
    flat_e_windows = torch.cat(all_energy_windows, dim=0)                      # (sum_L, WINDOW+1)

    # Phase 4: regression model in mini-batches
    disorder_chunks = []
    for i in range(0, flat_e_windows.shape[0], window_batch_size):
        disorder_chunks.append(
            regression_model(flat_e_windows[i:i + window_batch_size]).detach().cpu().numpy()
        )
    flat_disorder = np.concatenate(disorder_chunks)                            # (sum_L,)

    # Phase 5: split back by sequence, optionally smooth
    results = []
    offset = 0
    for L in lengths:
        d = flat_disorder[offset:offset + L]
        offset += L
        if not no_smoothing and L >= 10:
            d = savgol_filter(d, 11, 5)
        results.append(d)
    return results


def low_memory_predict(sequence, embedding_model, decoder_model, device, no_smoothing=False, chunk_len=1000):
    overlap = 100
    if (len(sequence)-1) % (chunk_len-overlap) == 0:
        logging.warning('Chunk length decreased by 1 to fit sequence length')
        chunk_len -= 1
    if chunk_len <= overlap:
        raise ValueError("Chunk len must be bigger than 200!")
    overlapping_predictions = []
    for chunk in range(0, len(sequence), chunk_len-overlap):
        overlapping_predictions.append(predict_disorder(
            sequence[chunk:chunk+chunk_len],
            embedding_model,
            decoder_model,
            device,
            no_smoothing
        ))
    prediction = np.concatenate((overlapping_predictions[0], *[x[overlap:] for x in overlapping_predictions[1:]]))
    return prediction


def aiupred_disorder(sequence, force_cpu=False, gpu_num=0):
    """
    Library function to carry out single sequence analysis
    :param sequence: Amino acid sequence in a string
    :param force_cpu: Force the method to run on CPU only mode
    :param gpu_num: Index of the GPU to use, default=0
    :return: Numpy array with disorder propensities for each position
    """
    embedding_model, reg_model, device = init_models(force_cpu, gpu_num)
    return predict_disorder(sequence, embedding_model, reg_model, device)


def process_single_pdb_file(pdb_file, embedding_model, reg_model, device, three_to_one):
    """
    Process a single PDB file and return the results.
    
    Args:
        pdb_file (str): Path to the PDB file
        embedding_model: The embedding model
        reg_model: The regression model
        device: Device to run on
        three_to_one (dict): Dictionary mapping three-letter to one-letter amino acid codes
        
    Returns:
        dict: Results for the protein or None if error
    """
    try:
        accession_id = pdb_file.split(".")[0].split("-")[1]
        
        parser = PDBParser(PERMISSIVE=1)
        structure = parser.get_structure(accession_id, pdb_file)

        # Get the first model and chain (AlphaFold PDBs typically have only one model and chain)
        model = structure[0]
        chain = model["A"]

        # Extract sequence and B-factors
        sequence = []
        plddt = []

        for residue in chain:
            # Get the residue's one-letter code
            sequence.append(three_to_one[residue.resname])
            
            # Get CA atom's B-factor (using CA as it's present in all amino acids)
            ca_atom = residue['CA']
            plddt.append(ca_atom.get_bfactor())

        sequence = ''.join(sequence)
        disorder_score = predict_disorder(sequence, embedding_model, reg_model, device, no_smoothing=True)
        order_mask = np.array(disorder_score) <= 0.5
        plddt = np.array(plddt) / 100
        mean_plddt = np.mean(plddt)
        masked_mean_plddt = np.mean(plddt[order_mask])
        
        return {
            'accession_id': accession_id,
            'sequence': sequence,
            'disorder_score': str(disorder_score.tolist()),
            'plddt': str(plddt.tolist()),
            'mean_plddt': mean_plddt,
            'masked_mean_plddt': masked_mean_plddt
        }
        
    except Exception as e:
        print(f"Error processing {pdb_file}: {e}")
        return None


def aiupred_masked_mean_plddt(pdb_dir, result_file, embedding_model, reg_model, device, num_cpus=None, extract_accession_id=False):
    """
    Process the PDB files in the given directory and save the results to a CSV file.
    
    Args:
        pdb_dir (str): The directory containing the PDB files.
        result_file (str): The path to the CSV file to save the results.
        embedding_model: The embedding model
        reg_model: The regression model
        device: Device to run on
        num_cpus (int, optional): Number of CPUs to use for multiprocessing. 
                                 If None and device is CPU, uses all available CPUs.
    """
    # Convert three-letter codes to one-letter codes
    three_to_one = {
        'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F',
        'GLY': 'G', 'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L',
        'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 'ARG': 'R',
        'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'
    }
    
    pdb_files = glob(os.path.join(pdb_dir, "*.pdb"))
    print("Number of PDB files:", len(pdb_files))

    if os.path.exists(result_file):
        results_df = pd.read_csv(result_file)
        # Get list of already processed accession IDs
        processed_ids = set(results_df['accession_id'].values)
        # Filter out already processed files
        if extract_accession_id:
            pdb_files = [f for f in pdb_files if f.split(".")[0].split("-")[1] not in processed_ids]
        else:
            pdb_files = [f for f in pdb_files if f.split(".")[0] not in processed_ids]
        print(f"Found {len(pdb_files)} new files to process")
    else:
        results_df = pd.DataFrame(columns=['accession_id', 'sequence', 'disorder_score', 
                                        'plddt', 'mean_plddt', 'masked_mean_plddt'])
        processed_ids = set()
    
    if not pdb_files:
        print("No new files to process")
        return
    
    # Check if we should use multiprocessing
    use_multiprocessing = device == 'cpu' and len(pdb_files) > 1
    
    if use_multiprocessing:
        # Set number of CPUs
        if num_cpus is None:
            num_cpus = mp.cpu_count()
        else:
            num_cpus = min(num_cpus, mp.cpu_count())
        
        print(f"Using multiprocessing with {num_cpus} CPUs")
        
        # Create a partial function with fixed arguments
        process_func = partial(process_single_pdb_file, 
                             embedding_model=embedding_model, 
                             reg_model=reg_model, 
                             device=device, 
                             three_to_one=three_to_one)
        
        # Process files in parallel
        with mp.Pool(processes=num_cpus) as pool:
            results = list(tqdm(pool.imap(process_func, pdb_files), 
                              total=len(pdb_files), 
                              desc="Processing PDB files"))
        
        # Filter out None results (errors)
        valid_results = [r for r in results if r is not None]
        
        if valid_results:
            # Convert results to DataFrame
            new_results_df = pd.DataFrame(valid_results)
            results_df = pd.concat([results_df, new_results_df], ignore_index=True, axis=0)
            
    else:
        # Sequential processing (original method)
        print("Using sequential processing")
        for pdb_file in tqdm(pdb_files):
            result = process_single_pdb_file(pdb_file, embedding_model, reg_model, device, three_to_one)
            if result is not None:
                new_row_df = pd.DataFrame([result])
                results_df = pd.concat([results_df, new_row_df], ignore_index=True, axis=0)
                
                # Save after every 100 proteins (you can adjust this number)
                if len(results_df) % 100 == 0:
                    results_df.to_csv(result_file, index=False)
    
    # Format the results
    if len(results_df) > 0:
        results_df["disorder_score"] = results_df["disorder_score"].apply(
            lambda x: str(np.around(np.array(eval(x)), 4).tolist()))
        results_df["plddt"] = results_df["plddt"].apply(
            lambda x: str(np.around(np.array(eval(x)), 4).tolist()))

    results_df.to_csv(result_file, index=False)
    print(f"Results saved to {result_file}")


def main(multifasta_file, force_cpu=False, gpu_num=0, no_smoothing=False, low_memory=None):
    """
    Main function to be called from aiupred.py
    :param multifasta_file: Location of (multi) FASTA formatted sequences
    :param force_cpu: Force the method to run on CPU only mode
    :param gpu_num: Index of the GPU to use, default=0
    :return: Dictionary with parsed sequences and predicted results
    """
    embedding_model, reg_model, device = init_models(force_cpu, gpu_num)
    sequences = multifasta_reader(multifasta_file)
    logging.debug("Sequences read")
    logging.info(f'{len(sequences)} sequences read')
    if not sequences:
        raise ValueError("FASTA file is empty")
    results = {}
    logging.StreamHandler.terminator = ""
    for num, (ident, sequence) in enumerate(sequences.items()):
        if len(sequence) <= 10:
            logging.warning(f'Sequence length of {ident} is smaller than 10, smoothing will be turned off!\n')
        results[ident] = {}
        if low_memory:
            results[ident]['aiupred'] = low_memory_predict(sequence, embedding_model, reg_model, device, no_smoothing, low_memory)
        else:
            results[ident]['aiupred'] = predict_disorder(sequence, embedding_model, reg_model, device, no_smoothing)
        results[ident]['sequence'] = sequence
        logging.debug(f'{num}/{len(sequences)} sequences done...\r')

    logging.StreamHandler.terminator = '\n'
    logging.debug(f'Analysis done, writing output')
    return results