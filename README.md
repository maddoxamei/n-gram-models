# Performance Analysis of N-gram Language Models
Updated 10/9/2022

Introduction
----------
This code was developed for Assignment 1 of CSCI3450 at New College of Florida, Fall 2022 semester.

The goal is to compare the performance of different n-gram models, both smoothed and unsmoothed, on a set of test data.

The training & test sets are from the 20056 DUC dataset, (https://duc.nist.gov/duc2005/tasks.html), although they were not drawn from the original URL for the assignment.

Requirements
----------
Python libraries required:
	os,nltk,string,glob,re,typing,collections,functools,math,json

Training Set should be placed in a directory ..\data\TrainingSet\
	\TrainingSet\ should contain subfolders, which contain text files for each article.

Test Set should be placed in adjacent directory ..\data\TestSet\
	\TestSet\ should contain subfolders, which contain text files for each article.

Make sure the TestSet directory can fit a .json containing ngram dictionaries.

Running the Code
----------
Experiment can be run using the following in the command line:

python main.py -d ..\data\TestSet\ RETRAIN -d ..\data\TrainingSet\ -pt 0.2 -ct=5
    Creates n-gram dictionaries from training set.  Saves results in a .json in the TestSet directory.
python .\main.py -d ..\data\TestSet\  LOAD -j .\training_corpus.json
    Loads existing model.

Code Summary
----------
utils.py - Contains all functions called by main.py.
    get_corpus() - Compiles a corpus from documents.
        _get_tokenized_sentences() - takes a filename as a string, and returns a list of tokenized sentences
            _tokenize_sentence() - Preprocesses/tokenizes sentences.  Applies lemmatization.
                _regex_cleanup() - Removes undesirable headers, tags, etc.
    make_corpus() - Generates vocabulary/frequency counts for an n-gram (1-4) from a given corpus, and saves as a .json.
        get_grams() creates n-gram dictionaries from the input data
            __add_pseudo_words() - Adds sentence start/end tokens (<s>,</s>) for non-unigram n-grams.
            __trans_pseudo_words() - Adds start/end tokens for sentence transitions.
            _create_unknowns() - Creates unknown (<unk>) tags based on token frequency.
    print_metrics() - Prints likelihood & perplexity metrics for set of test data.
        predict() is used for calculating log likelihood & perplexity of a sentence, using an ngram model. Returns (likelihood,perplexity) as a tuple.	
            _log_MLE() - is used exclusively in predict() for calculating likelihood.
main.py - Main file for running the experiment.  Includes a parser for running from command line.

