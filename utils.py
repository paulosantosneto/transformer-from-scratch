import torch
import torchdata.datapipes as dp
import torchtext.transforms as T
import spacy
import argparse
import matplotlib.pyplot as plt
import json
import os
import numpy as np

from torchtext.vocab import build_vocab_from_iterator, vocab
from typing import Tuple, List

def get_args():

    parser = argparse.ArgumentParser()
    
    # --- train ---

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default='123', type=str)
    parser.add_argument('--test_size', default=0.2, type=float)    
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--file_path', default='data/machine_translation/teste.tsv', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--dm', default=512, type=str)
    parser.add_argument('--batch_size', default=64, type=str)
    parser.add_argument('--N', default=1, type=int)
    parser.add_argument('--dff', default=2048, type=int)
    parser.add_argument('--heads', default=8, type=int)
    parser.add_argument('--save', default=True, type=bool)
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    parser.add_argument('--max_len', default=64, type=int)
    parser.add_argument('--min_freq', default=2, type=int)

    # --- inference ---
    
    parser.add_argument('--model_load_path', default=None, type=str)
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--source_sentence', nargs='+', default="I lost my inspiration.", type=str)
    parser.add_argument('--source_vocab', default='weights/en_vocab.pth', type=str)
    parser.add_argument('--target_vocab', default='weights/pt_vocab.pth', type=str)
    parser.add_argument('--source_tokenizer', default='en_core_web_sm', type=str)
    parser.add_argument('--target_tokenizer', default='pt_core_news_sm', type=str)

    args = parser.parse_args()

    if args.mode == 'inference' and (args.model_load_path is None or args.model_config_path is None):
        parser.error('--model_load_path and model_config_path is mandatory when --mode is "inference"')

    return args

def load_and_preprocessing_data(args: dict):
    """
    Receives a FILE_PATH with entire sentences for translation and return
    a DataLoader with train/test datasets.
    
    Steps: 1.Open File 2.Tokenize sentences 3.Build Vocab 4.Add special tokens 
           5.Numerizalize 6.Batches 7.Padding
    """
    
    global eng, pt, en_vocab, pt_vocab
    
    FILE_PATH = args.file_path
    test_size = args.test_size
    BATCH_SIZE = args.batch_size
    max_length = args.max_len
    source_tokenizer = args.source_tokenizer
    target_tokenizer = args.target_tokenizer

    eng = spacy.load(source_tokenizer)
    pt = spacy.load(target_tokenizer)

    datapipe = dp.iter.IterableWrapper([FILE_PATH])
    datapipe = dp.iter.FileOpener(datapipe, mode='rb')
    datapipe = datapipe.parse_csv(skip_lines=0, delimiter='\t', as_tuple=True)
    datapipe = datapipe.map(removeAttribute)
    
    DATASET_SIZE = len(list(datapipe))
    
    special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']

    # vocab of english sentences
    en_vocab = build_vocab_from_iterator(
        getTokens(datapipe, 0),
        min_freq=args.min_freq,
        specials=special_tokens,
        special_first=True
    )

    en_vocab.set_default_index(en_vocab['<unk>'])
    
    # vocab of portuguese sentences
    pt_vocab = build_vocab_from_iterator(
        getTokens(datapipe, 1),
        min_freq=args.min_freq,
        specials=special_tokens,
        special_first=True
    )

    pt_vocab.set_default_index(pt_vocab['<unk>'])
    
    datapipe = datapipe.map(applyTransform)
    
    datapipe = datapipe.bucketbatch(
        batch_size = BATCH_SIZE, 
        batch_num = (DATASET_SIZE + BATCH_SIZE - 1) // BATCH_SIZE,
        bucket_num=1,
        use_in_batch_shuffle=False
    )
    
    datapipe = datapipe.map(separateLanguages)
    
    datapipe = datapipe.map(lambda x: applyPadding(x, max_length))
    
    train, test = datapipe.random_split(total_length=DATASET_SIZE, weights={'train': (1 - test_size),
    'test': test_size}, seed=0)

    return train, test, en_vocab, pt_vocab, eng, pt

def applyPadding(pair_of_sequences, max_len):
    """
    Convert sequences to tensors and apply padding.
    
    input of form: [[(X_1, ..., X_n), (y_1, ..., y_n)]_1, ..., [(X_1, ..., X_n), (y_1, ..., y_n)]_b], where 'n' is the batch_size and 'b' is the number of batches.

    output: transform to tensor and apply special token '<pad>' (position  0) to padding the shortest sentences.
    """
    
    # Convert tuples to lists and pad the sequences with '0' tokens to match the max length
    padded_sequences_source = [list(seq)[1:] + [0] * (max_len - len(seq) + 1) for seq in pair_of_sequences[0]]
    padded_sequences_target = [list(seq) + [0] * (max_len - len(seq)) for seq in pair_of_sequences[1]]
    
    shifted_sequences_target = [list(seq)[1:] + [0] * (max_len - len(seq) + 1) for seq in pair_of_sequences[1]]

    # Convert the padded sequences to tensors
    input_target = T.ToTensor(0)(padded_sequences_target)
    input_source = T.ToTensor(0)(padded_sequences_source)
    output_target = T.ToTensor(0)(shifted_sequences_target)

    return (input_source, input_target, output_target)

def separateLanguages(sequence_pairs):
    """
    input of form: [(X_1, y_1), ..., (X_n, y_n)]
    output of form: [(X_1, ..., X_n), (y_1, ..., y_n)]
    """
    sources, targets = zip(*sequence_pairs)

    return sources, targets

def applyTransform(sequence_pair) -> List:
    """
    Apply transforms to sequence of tokens in a sequence pair.
    """
    
    return (
        getTransform(en_vocab)(engTokenize(sequence_pair[0])),
        getTransform(pt_vocab)(ptTokenize(sequence_pair[1]))
    )

def sortBucket(bucket):
    
    return sorted(bucket, key=lambda x: (len(x[0]), len(x[1])))

def getTransform(vocab):
    """
    Transformation of words to indices.
    """
    text_transform = T.Sequential(
        T.VocabTransform(vocab=vocab),
        T.AddToken(1, begin=True), # <sos> token
        T.AddToken(2, begin=False)
    )

    return text_transform

def getTokens(dataiter, place):
    """
    Function to yield tokens from an iterator.
    """
    for english, portuguese in dataiter:
        if place == 0:
            yield engTokenize(english)
        else:
            yield ptTokenize(portuguese)

def removeAttribute(row: Tuple) -> Tuple:
    """
    Filters only the columns with texts.
    """
    return row[1:4:2] 

def engTokenize(text: str) -> List:
    """
    Tokenize an English text and return a list of tokens.
    """
    return [token.text for token in eng.tokenizer(text)]

def ptTokenize(text: str) -> List:
    """
    Tokenize a Portuguese text and return a list of tokens.
    """
    return [token.text for token in pt.tokenizer(text)]

def inference_preprocessing(configs, args, source_vocab, target_vocab, source_tokenizer, target_tokenizer):

    source_tokens = [token.text for token in source_tokenizer.tokenizer(args.source_sentence)]
    
    source = [source_vocab.get_stoi()['<sos>']]
    target = [target_vocab.get_stoi()['<sos>']]

    source.extend([source_vocab.get_stoi()[token] for token in source_tokens])
    source.append(source_vocab.get_stoi()['<eos>'])
    
    source = [source + [source_vocab.get_stoi()['<pad>']] * (configs['max_len'] - len(source))]
    target = [target + [target_vocab.get_stoi()['<pad>']] * (configs['max_len'] - len(target))]

    return T.ToTensor(0)(source), T.ToTensor(0)(target)

def plot_loss(epochs: int, loss_histories: list, labels: list):
    plt.clf()

    plt.plot(np.arange(epochs), loss_histories, label=labels)

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')
    plt.clf()

# --- Load Model ---

def save_model(model: any, epochs: str, architecture: str):

    try:
        if not os.path.exists('weights'):
            os.makedirs('weights')

        torch.save(model.state_dict(), f'weights/best_model_{architecture}_{epochs}.pth')

        return 'Model was successfully saved!'
    except:
        raise 'There was a problem saving the model!'

def save_configs(en_vocab, pt_vocab, args: dict, model_name: str):
    
    torch.save(en_vocab, args.source_vocab)
    torch.save(pt_vocab, args.target_vocab)
    
    with open(f'weights/configs_{model_name}.json', 'w') as json_config:
        json.dump(vars(args), json_config)

def load_configs(args: dict) -> dict:
    
    configs = {}

    with open(f'weights/{args.model_config_path}', 'r') as json_config:
        configs = json.load(json_config)
    
    eng = spacy.load(args.source_tokenizer)
    pt = spacy.load(args.target_tokenizer)
    
    en_vocab = torch.load(args.source_vocab)
    pt_vocab = torch.load(args.target_vocab)

    return configs, eng, pt, en_vocab, pt_vocab

