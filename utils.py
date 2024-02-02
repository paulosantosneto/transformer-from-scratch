import torchdata.datapipes as dp
import torchtext.transforms as T
import spacy
import argparse

from torchtext.vocab import build_vocab_from_iterator
from typing import Tuple, List

def get_args():

    parser = argparse.ArgumentParser()
    
    # --- train ---

    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--seed', default='123', type=str)
    parser.add_argument('--test_size', default=0.2, type=float)    
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--path_file', default='data/machine_translation/tatoeba_en_pt.tsv', type=str)
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--dim', default=512, type=str)
    parser.add_argument('--save_model', default=True, type=bool)
    parser.add_argument('--plot_loss', action='store_true')
    parser.add_argument('--verbose', action='store_true')

    # --- inference ---
    
    parser.add_argument('--model_load_path', default=None, type=str)
    parser.add_argument('--model_config_path', default=None, type=str)
    parser.add_argument('--source_sentence', nargs='+', default='Hello World!', type=str)

    args = parser.parse_args()

    if args.mode == 'inference' and (args.model_load_path is None or args.model_config_path is None):
        parser.error('--model_load_path and model_config_path is mandatory when --mode is "inference"')

    return args

def load_and_preprocessing_data(FILE_PATH: str, test_size: float):
    """
    Receives a FILE_PATH with entire sentences for translation and return
    a DataLoader with train/test datasets.
    
    Steps: 1.Open File 2.Tokenize sentences 3.Build Vocab 4.Add special tokens 
           5.Numerizalize 6.Batches 7.Padding
    """
    
    global eng, pt, en_vocab, pt_vocab

    eng = spacy.load('en_core_web_sm')
    pt = spacy.load('pt_core_news_sm')

    datapipe = dp.iter.IterableWrapper([FILE_PATH])
    datapipe = dp.iter.FileOpener(datapipe, mode='rb')
    datapipe = datapipe.parse_csv(skip_lines=0, delimiter='\t', as_tuple=True)
    datapipe = datapipe.map(removeAttribute)
    
    DATASET_SIZE = len(list(datapipe))
    
    special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']

    # vocab of english sentences
    en_vocab = build_vocab_from_iterator(
        getTokens(datapipe, 0),
        min_freq=2,
        specials=special_tokens,
        special_first=True
    )

    en_vocab.set_default_index(en_vocab['<unk>'])
    
    # vocab of portuguese sentences
    pt_vocab = build_vocab_from_iterator(
        getTokens(datapipe, 1),
        min_freq=2,
        specials=special_tokens,
        special_first=True
    )

    pt_vocab.set_default_index(pt_vocab['<unk>'])
    
    datapipe = datapipe.map(applyTransform)
    
    BATCH_SIZE = 32

    datapipe = datapipe.bucketbatch(
        batch_size = BATCH_SIZE, 
        batch_num = (DATASET_SIZE + BATCH_SIZE - 1) // BATCH_SIZE,
        bucket_num=1,
        use_in_batch_shuffle=False,
        sort_key=sortBucket
    )
    
    datapipe = datapipe.map(separateLanguages)
    
    datapipe.map(applyPadding)
    
    train, test = datapipe.random_split(total_length=DATASET_SIZE, weights={'train': (1 - test_size),
    'test': test_size}, seed=0)

    return train, test

def applyPadding(pair_of_sequences):
    """
    Convert sequences to tensors and apply padding.
    
    input of form: [[(X_1, ..., X_n), (y_1, ..., y_n)]_1, ..., [(X_1, ..., X_n), (y_1, ..., y_n)]_b], where 'n' is the batch_size and 'b' is the number of batches.

    output: transform to tensor and apply special token '<pad>' (position 0) to padding the shortest sentences.
    """

    return (T.ToTensor(0)(list(pair_of_sequences[0])), T.ToTensor(0)(list(pair_of_sequences[1])))

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

