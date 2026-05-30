import argparse
import logging
import aiupred_lib

parser = argparse.ArgumentParser(
    description='AIUPred disorder prediction method v0.9\n'
                'Developed by Zsuzsanna Dosztanyi and Gabor Erdos',
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument("-i", "--input_file",
                    help="Input file in (multi) FASTA format",
                    required=True, type=argparse.FileType('r', encoding='UTF-8'))
parser.add_argument("-o", "--output_file",
                    help="Output file")
parser.add_argument("-v", "--verbose",
                    help="Increase output verbosity",
                    action="store_true")
parser.add_argument("-g", "--gpu",
                    help="Index of GPU to use, default=0",
                    default=0)
parser.add_argument("--force-cpu",
                    help="Force the network to only utilize the CPU. Calculation will be very slow, not recommended",
                    action="store_true")
parser.add_argument("--no-smoothing",
                    help="Removes the default SavGol smoothing function",
                    action="store_true")
parser.add_argument("--low-memory", nargs='?',
                    help="Use chunking to lower the memory usage. Default chunk size is 1000. The lower the chunk size the lower the memory consumption well as the accuracy",
                    const=1000, type=int)

args = parser.parse_args()

if args.verbose:
    logging.basicConfig(level=logging.DEBUG, format='# %(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


output_str = '''#             _____ _    _ _____              _ 
#       /\   |_   _| |  | |  __ \            | |
#      /  \    | | | |  | | |__) | __ ___  __| |
#     / /\ \   | | | |  | |  ___/ '__/ _ \/ _` |
#    / ____ \ _| |_| |__| | |   | | |  __/ (_| |
#   /_/    \_\_____|\____/|_|   |_|  \___|\__,_|
#
# Version 1.2.2
# AIUPred: combining energy estimation with deep learning for the enhanced prediction of protein disorder
# Gabor Erdos, Zsuzsanna Dosztanyi
# Nucleic Acid Research 2024 gkae385
# https://doi.org/10.1093/nar/gkae385'''
print(output_str)
if not args.output_file:
    output_str = ''
for ident, results in aiupred_lib.main(args.input_file,
                                       force_cpu=args.force_cpu,
                                       gpu_num=args.gpu, no_smoothing=args.no_smoothing,
                                       low_memory=args.low_memory).items():
    output_str += ident + '\n'
    for pos, value in enumerate(results['aiupred']):
        output_str += f'{pos+1}\t{results["sequence"][pos]}\t{value:.4f}\n'
    output_str += '\n\n'
logging.info('Analysis done, writing output')
if args.output_file:
    with open(args.output_file, 'w') as file_handler:
        file_handler.write(output_str.strip())
else:
    print(output_str.strip())
