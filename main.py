import argparse
import random
from utils import *

parser = argparse.ArgumentParser(description='Evaluate an n-gram model')
parser.add_argument('--test_dir', '-d', type=str, required=True, help="Path to test documents")
parser.add_argument('--test_recursive', '-r', action='store_true', help="Flag to indicate data is stored in a recursive file structure")
parser.add_argument('--number', '-n', type=int, default=None, help="Number of randomly sampled sentences to calculate model average log likelihood and perplexity on. Default is to use all sentences")
parser.add_argument('--k', '-k', type=float, default=1.0, help="Probability mass  to move from the seen to unseen events; used in add-k smoothing") 
parser.add_argument('--weight', '-w', type=float, default=0.4, help="Weight of lower order n-gram when backing off") 
parser.add_argument('--standardization_method', '-s', type=str, default='lemmatize', choices=['stem', 'lemmatize'], help="Standardization method (default: %(default)s)")
subparsers = parser.add_subparsers(help="Positional Argument (stored in cmd) to determine training data", dest="cmd", required=True)
load_parser = subparsers.add_parser('LOAD')
load_parser.add_argument('--train_json', '-j', type=str, required=True, help="Filepath of training corpus .json")
retrain_parser = subparsers.add_parser('RETRAIN')
retrain_parser.add_argument('--train_dir', '-d', type=str, required=True, help="Path to training documents")
retrain_parser.add_argument('--prob_thresh', '-pt', type=float, default=0.0, help="Probability of the least common unigrams to convert to unknown tokens")
retrain_parser.add_argument('--count_thresh', '-ct', type=int, default=0, help="Unigrams with raw frequency counts below this integer threshold value are converted to unknowns") 
retrain_parser.add_argument('--train_recursive', '-r', action='store_true', help="Flag to indicate data is stored in a recursive file structure")

if __name__ == '__main__':
    args = parser.parse_args()
    # Load training corpus if applicable, otherwise generate one
    if args.cmd == 'LOAD' and os.path.exists(args.train_json):
        print("Loading training corpus...")
        with open(args.train_json, 'r') as file:
            grams = json.load(file)
        if isinstance(list(grams.keys())[0], str):
            [grams.update({i:grams.pop(str(i))}) for i in range(1,5)]
    else:
        print("Generating training corpus...")
        grams = make_corpus(args.train_dir, args.train_recursive, threshold=args.count_thresh, prop_thresh=args.prob_thresh)
        
    # Substitute unknown token for out of vocabulary words in loaded test corpus
    corpus = []
    unknowns = 0
    others = 0
    for sentence in get_corpus(args.test_dir, args.test_recursive, standardization_method=globals()[args.standardization_method]):
        s = []
        for word in sentence:
            if grams.get(1).get(word) is None:
                s.append('<ukn>')
                unknowns += 1
            else:
                s.append(word)
                others += 1
        corpus.append(s)

    # Print test corpus statistics
    print("# Sentences:", len(corpus), 
          "\n# Uknowns:", unknowns,
          "\nTotal Tokens:", others+unknowns,
          "\nProportion of Uknowns:", unknowns / (others+unknowns))
    
    if args.number is not None:
        corpus = random.sample(corpus, args.number)
        with open('sentences.txt', 'w') as file:
            print(*corpus,sep="\n",file=file)

    for n in range(2,5):
        # gram statistics
        print(f'{n}-grams:\n', 
          f'\n\t# of unique {n}-grams: {len(grams.get(n))}')
        
        # n-gram model without smoothing
        likelihoods, token_counts = zip(*[predict(s, n, grams, k=args.k, lambda_=args.weight, method=backoff) for s in corpus])
        print("(Without Smoothing, Backoff)")
        print_metrics(likelihoods, token_counts, n, grams, k=args.k, lambda_=args.weight)
        
        # n-gram model with basic smoothing if 0-count
        likelihoods, token_counts = zip(*[predict(s, n, grams, k=args.k, lambda_=args.weight) for s in corpus])
        print("(Without Smoothing, Basic)")
        print_metrics(likelihoods, token_counts, n, grams, k=args.k, lambda_=args.weight)
        
        # n-gram model with smoothing
        likelihoods, token_counts = zip(*[predict(s, n, grams, k=args.k, lambda_=args.weight, smoothing=True) for s in corpus])
        print("(With Smoothing)")
        print_metrics(likelihoods, token_counts, n, grams, k=args.k, lambda_=args.weight)