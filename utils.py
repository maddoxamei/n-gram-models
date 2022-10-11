import os
import nltk
import string
from glob import iglob
import re
from typing import List, Dict, Tuple
from collections import Counter
from functools import reduce
import math
import json

def _regex_cleanup(sentence: str) -> str:
    """
    Basic sentence preprocessing to remove punctuation, convert particular types of text into pseudo words, and handle edge-cases present in the training/test corpus. Cleanup process does NOT include token/word standardization.
    @param sentence: sequence of words to clean up
    @return: cleaned sequence of words
    """
    # Convert hyperlinks into pseudo word
    sentence = re.sub(r"http[s]?://(?:\w+[.|/]?)+", '<hyp>', sentence)
    # Convert decimals or floating points to pseudo word
    # sentence = re.sub(r"\d*[.]\d+", '<dec>', sentence)
    # Convert emails into pseudo word
    sentence = re.sub(r"(?:\w+[.]?)+@(?:\w+[.]?)+", '<email>', sentence)
    # substitute newline character and "p" tags with space
    # separate hyphenated words (e.g. mid-1990 or 2pm-4pm)
    sentence = re.sub(r'<[/]*p>|\n|-', ' ', sentence)
    # Separate num. from letters (e.g. "2PM" to "2 PM")
    sentence = re.sub('(\d)(\D)', r'\1 \2', sentence)
    # Convert standalone 4-digit non-character separated num. into pseudo word (e.g. convert "4564" but not "34.34", "34:34", nor "12345")
    # sentence = re.sub('\s\d{4}[.]?\s', '<4num>', sentence)
    # Remove non \' punctuation that is not associated with a pseudo word
    sentence = re.sub(r"(?<!<hyp)(?<!<dec)(?<!<email)(?<!<4num)[^\w\s\'](?!hyp>|dec>|email>|4num>)", '', sentence)
    # Remove non conjunctive \'
    sentence = re.sub(r"(?<!\w)\'|\'\s", '', sentence)
    return sentence

stem = lambda x, tag: nltk.stem.PorterStemmer().stem(x)
lemmatize = lambda x, tag: nltk.stem.WordNetLemmatizer().lemmatize(x, pos=tag)

def _tokenize_sentence(sentence: str, standardization_method=lemmatize) -> List[str]:
    """ Preprocess and tokenize a given sentence. Preprocessign includes token standardization.
    @param sentence: sentence to preprocess and then tokenize
    @param standardization_method: either stem or lemmatize
    @return: preprocessed sentence
    """
    # Clean up the text
    sentence = nltk.pos_tag(nltk.word_tokenize(_regex_cleanup(sentence)), tagset='universal')
    # Skip uncessary work; sometimes the cleaning process removes all characters (e.g. "!!" to "")
    if sentence == []:
        return sentence
    
    # The lemmatizer standardizes a word differently depending on the part of speech
    # The dictionary acts as an interface for the pos_tag() output to the lemmatize() input
    tokens, tags = zip(*sentence)
    ref = {'ADV':'s', 'ADJ':'r', 'VERB':'v', 'NOUN':'n','PRON':'n', 'PROPN':'n'}
    
    i = 0
    sentence = []
    while i < len(tokens):
        # Tokenization process splits pseudo words (e.g. "<hyp>" to ["<", "hyp", ">"])
        # Rejoin pseudo words (e.g. ["<", "hyp", ">"] back to "<hyp>")
        if tokens[i] == '<':
            sentence.append(''.join(tokens[i:i+3]))
            i += 3
        # Standardize the non-pseudo words in the sentence (e.g. "dogs" to "dog")
        else:
            sentence.append(standardization_method(tokens[i], ref.get(tags[i], 's')))
            i += 1
    return sentence
    
def get_tokenized_sentences(filename: str, **kwargs) -> List[List[str]]:
    """ Pre-process and tokenize each sentence in the given document 
        @param filename: file path to a particular corpus document 
        @return: preprocessed sentences from the given document
    """
    with open(filename, 'r', encoding='utf-8') as file:
        # lowercase all text in the document
        text = file.read().lower()
        # extract the text between the "text" tags
        text = re.findall(r"<text>([\s\S.]*)</text>", text)
        if text == []:
            return text
    # remove text from tables; assumes tables denoted by at least 4 hyphens or the table tag
    text = re.sub(r"<table([\s\S.]*)>([\s\S.]*)</table>|[-]{4,}([\s\S.]*)[-]{4,}|-", ' ', text[0])
    
    # Identify sentences via "\n.", "!", "?", ";" in addition to the contextually Identify sentences via "."
    tokenizer = nltk.RegexpTokenizer('(?:[.]{1}\n)|[!?;]', gaps=True)
    sentences = sum([tokenizer.tokenize(s) for s in nltk.sent_tokenize(text)], [])
    
    # remove further cleaning, tokenization, and lemmatization of the words in each sentence
    return [s for s in (_tokenize_sentence(s, **kwargs) for s in sentences) if s != []]

def _add_pseudo_words(sentence: List[str], n: int) -> List[str]:
    """Adds beginning and end of sentence pseudo words <s> or </s>, respectively, to allow calculation of sentence probability. The number of pseudo words added depends on the n-gram.
        @param sentence: Tokenized sentence to add pseudo words to
        @param n: length of the n-gram
        @return tokenized sentence with beginning and closing pseudo words prepended and appended, respectively
    """
    return ['<s>']*(n-1) + sentence + ['</s>']*(n-1)

def _trans_pseudo_words(n: int) -> List[str]:
    """
    Mimics the tokens involved in a sentence transition for a particular n-gram (e.g. </s><s> for bigram)
    @param n: length of the n-gram
    """
    return ['</s>']*(n-1) + ['<s>']*(n-1)

def _create_unknowns(tokenized_sentences: List[str], threshold: int = 0, prop_thresh: float = 0.0, **kwargs): 
    """
    Converts particular tokens into uknown tags based on frequency count and/or existing below a particular percentile
    @param tokenized_sentences:
    @param threshold:
    @prop_thresh:
    """
    if threshold == 0 and prop_thresh == 0.0:
        return tokenized_sentences
    counter = reduce(Counter.__add__, [Counter(s) for s in tokenized_sentences])
    print("Unique unigrams before unknownization:", len(counter))
    # Drop the prop_thresh least common tokens
    counter = dict(counter.most_common(round(len(counter)*(1-prop_thresh))))
    # Convert all tokens in the sentence which were dropped or have frequencies below the threshold into unknown tokens
    return [['<ukn>' if counter.get(w, 0) - threshold <= 0 else w for w in s] for s in tokenized_sentences]

def get_grams(tokenized_sentences: List[List[str]], **kwargs) -> Dict[int, Dict[str, int]]:
    """ Creates a n-gram dictionary for tokens in list of sentences
        @param tokenized_sentences: List of tokens for each sentence
        @return: unigram, bigram, trigram, and 4-gram dictionary
    """
    tokenized_sentences = _create_unknowns(tokenized_sentences, **kwargs)
    # List to hold the n-gram dictionaries
    gram_list = []
    for i in range(1,5): # Range ensures 1 to 4 n-grams created
        print(f"Starting {i}-gram creation")
        # Split each sentence into n-grams and caclulate the combined frequency totals across all sentences
        counter = dict(reduce(Counter.__add__, [Counter(nltk.ngrams(_add_pseudo_words(s, i), i)) for s in tokenized_sentences]))
        # Convert the tuple keys into a single string separated by a space.
        counter = dict([(' '.join(gram), count) for gram, count in counter.items()])
        
        # Adds a reference for pseudo words; used by the (n+1)-gram
        counter.update({' '.join(['<s>']*(i)):len(tokenized_sentences), # (e.g. P(I|<s>)
                       ' '.join(['</s>']*(i)):len(tokenized_sentences)}) # (e.g. P(</s>|work)
        # Adds a reference for sentence transitions; used by the (n+1)-gram (e.g. P(<s>|</s>)
        x = _trans_pseudo_words(i)
        counter.update(dict([(' '.join(x[k:k+i]), len(tokenized_sentences)-1) for k in range(i-1)]))

        # Print statistics
        print("\t# Unique n-grams:", len(counter))
        # Add the n-gram dictionary to 
        gram_list.append( counter )
    # Convert the list of n-gram dictionaries into a dictionary itself w/the key being n
    return dict(enumerate(gram_list, 1))

# Use stupid back-off strategey if zero-frequency n-gram
backoff = lambda tokens, grams, k, lambda_: lambda_*_log_MLE(tokens[1:], grams, False, k, lambda_, backoff)
# Use basic smoothing if zero-frequency n-gram
basic = lambda tokens, grams, k, lambda_: _log_MLE(tokens, grams, True, k, lambda_, basic)

def _log_MLE(tokensL List[str], grams: Dict[int, Dict[str, int]], smoothing: bool = False, k: float = 1.0, lambda_: float = 0.4, method=basic):
    """
    Calculates the log probability of a token given a history of tokens. Assumes the tokens from the beginning to penultimate tokens constitues the history. Unigrams (token list of length 1) assumes no history.
    @param tokens: sequence of tokens
    @param grams: reference vocabulary for n-grams (1-4)
    @param smoothing: whether to introduce Laplace-k smoothing
    @param k: amount of probability mass  to move from the seen to unseen events
    @param lambda_: scalar weight of lower order n-gram when backing off
    @param method: how to handle zero-frequency grams when Laplace-k smoothing not implemented
    """
    numerator = grams.get(len(tokens)).get(' '.join(tokens), 0) + (int(smoothing)*k)
    
    if numerator == 0:
        return method(tokens, grams, k, lambda_)
        # return lambda_*log_MLE(tokens[1:], grams, smoothing, k, lambda_)
        # return log_MLE(tokens, grams, True, k, lambda_, **kwargs)
    smoothed_vocab = (int(smoothing)*k) * len(grams.get(1)) # smoothing flag * vocab size (total unique tokens)
    if len(tokens) == 1:
        return math.log(numerator / (sum(grams.get(1).values()) + smoothed_vocab))
    return math.log(numerator / (grams.get(len(tokens)-1).get(' '.join(tokens[:-1]), 0) + smoothed_vocab))

def predict(sentence: List[str], n: int, grams: Dict[int, Dict[str, int]], **kwargs):
    """ Calculates the log probability of the entire sentence using chain probability.
    @param sentence: desired sentence to calculate the probability of
    @param n: desired length of n-gram
    @param grams: reference vocabulary for n-grams (1-4)
    """
    if sentence == []:
        return 0, 0, 0
    length = len(sentence)+n-1
    sentence = _add_pseudo_words(sentence, n)
    likelihoods = [_log_MLE(sentence[i:i+n], grams, **kwargs) for i in range(length)]
    return likelihoods, len(sentence)

def get_corpus(path, recursive, **kwargs):
    """ Compile a corpus from documents on file. Corpus is composed of sentences from all included documents.
    @param path: path to documents which constitute the desired corpus
    @param recursive: whether to look in subdirectory for additional documents
    @return: List of preprocessed and tokenized sentences from all the documents
    """
    # preprocess and tokenize sentences in each document
    files = [get_tokenized_sentences(file, **kwargs) for file in iglob(os.path.join(path, '**/*'), recursive=recursive) if os.path.isfile(file)]
    print( "# Documents:", len([file for file in files if file != []]) )
    # aggregate the sentences from each document 
    return sum(files, [])

def make_corpus(path, recursive, **kwargs):
    """ Generate the vocabulary and frequency counts for particular n-gram (1-4) sequences for a given corpus
    @param path: path to documents which constitute the desired corpus
    @param recursive: whether to look in subdirectory for additional documents
    @return: reference vocabulary for n-grams (1-4)
    """
    # Preprocess and tokenize sentences
    corpus = get_corpus(path, recursive)
    # Extract vocabulary and frequency counts
    grams = get_grams(corpus, **kwargs)
    # Save the reference vocabulary for quick/easy loading in later experiments
    with open(f'training_corpus.json', 'w') as file:
        json.dump(grams, file)
    # Report statistics
    print("# Sentences:", len(corpus), 
          "\n# Converted Uknowns:", grams.get(1).get('<ukn>'),
          "\nProportion of Converted Uknowns:", grams.get(1).get('<ukn>') / sum(grams.get(1).values()))
    return grams


def print_metrics(likelihoods, token_counts: List[int], n, grams: Dict[int, Dict[str, int]], **kwargs):
    """ Calculate the average sentence log likelihood and overall model perplexity
    @param likelihoods: log probability of each sentence
    @param token_counts: number of tokens in each sentence
    @param n: length of n-gram
    @param grams: reference vocabulary for n-grams (1-4) 
    """
    n_sentences = len(likelihoods)
    likelihoods = sum(likelihoods, [])
    print(f'\tAverage Log Likelihood: {sum(likelihoods)/n_sentences}')

    # Calculate the probability of sentence changes </s>...</s_n><s>...<s_n>
    likelihoods += predict(_trans_pseudo_words(n), n, grams, **kwargs)[0]
    token_counts = sum(token_counts)
    print('\tAverage Perplexity:', reduce(lambda x, y: x*y, [math.exp(l)**(-1/token_counts) for l in likelihoods]))
