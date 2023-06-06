import sys
import os
import string
import math
import nltk


FILE_MATCHES = 1
SENTENCE_MATCHES = 1

def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
        sys.exit("Usage: python questions.py corpus")

    # Calculate IDF values across files
    files = load_files(sys.argv[1])
    file_words = {
        filename: tokenize(files[filename])
        for filename in files
    }
    file_idfs = compute_idfs(file_words)

    # Prompt user for query
    query = set(tokenize(input("Query: ")))

    # Determine top file matches according to TF-IDF
    filenames = top_files(query, file_words, file_idfs, n=FILE_MATCHES)

    # Extract sentences from top files
    sentences = dict()
    for filename in filenames:
        for passage in files[filename].split("\n"):
            for sentence in nltk.sent_tokenize(passage):
                tokens = tokenize(sentence)
                if tokens:
                    sentences[sentence] = tokens

    # Compute IDF values across sentences
    idfs = compute_idfs(sentences)

    # Determine top sentence matches
    matches = top_sentences(query, sentences, idfs, n=SENTENCE_MATCHES)
    for match in matches:
        print(match)


def load_files(directory):
    """
    Given a directory name, return a dictionary mapping the filename of each
    `.txt` file inside that directory to the file's contents as a string.
    """
    mapping = dict()
    for file_name in os.listdir(directory):
        with open(os.path.join(directory, file_name)) as f:
            contents = f.read()
            mapping[file_name] = contents
    return mapping

def tokenize(document:str):
    """
    Given a document (represented as a string), return a list of all of the
    words in that document, in order.

    Process document by coverting all words to lowercase, and removing any
    punctuation or English stopwords.
    """
    # enforce lowercase
    document = document.lower()

    # remove punctuation
    document = document.translate({ord(i): None for i in string.punctuation})
    
    # tokenize
    words = nltk.word_tokenize(document)

    # remove stopwords
    for word in words.copy():
        if word in nltk.corpus.stopwords.words("english"):
            words.remove(word)

    return words


def calc_idf(docs_containing_word, num_docs):
    """
    calculates the idf of word with given statistics
    """
    return math.log(num_docs / docs_containing_word)


def num_docs_with_word(word, documents):
    """
    returns the sum of the number of documents that contain a given word
    """
    return sum(word in words for words in documents.values())


def get_all_words(documents):
    """
    returns a set of all words seen in documents
    """
    word_bag = set()
    for document in documents.values():
        for word in document:
            word_bag.add(word)
    return word_bag


def compute_idfs(documents):
    """
    Given a dictionary of `documents` that maps names of documents to a list
    of words, return a dictionary that maps words to their IDF values.

    Any word that appears in at least one of the documents should be in the
    resulting dictionary.
    """
    idf_map = dict()
    num_documents = len(documents)

    for word in get_all_words(documents):
        idf_map[word] = calc_idf(num_docs_with_word(word, documents), num_documents)
    return idf_map

def file_score(files, query, idfs):
    def internals(filename):
        """
        calculates the relevance score of the given file
        """
        sum = 0
        for word in query:
            if word in files[filename]:
                sum += idfs[word]
        return sum
    return internals


def top_files(query, files, idfs, n):
    """
    Given a `query` (a set of words), `files` (a dictionary mapping names of
    files to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the filenames of the the `n` top
    files that match the query, ranked according to tf-idf.
    """
    file_list = list(files)
    file_list.sort(key=file_score(files, query, idfs))
    # return top n of list
    return file_list[-n:]


def sentence_score(sentences, query, idfs):
    def internals(sentence):
        """
        calculates the relevance score of the given file
        """
        sum = 0
        for word in query:
            if word in sentences[sentence]:
                sum += idfs[word]
        return sum
    return internals


def top_sentences(query, sentences, idfs, n):
    """
    Given a `query` (a set of words), `sentences` (a dictionary mapping
    sentences to a list of their words), and `idfs` (a dictionary mapping words
    to their IDF values), return a list of the `n` top sentences that match
    the query, ranked according to idf. If there are ties, preference should
    be given to sentences that have a higher query term density.
    """
    sentence_list = list(sentences)
    sentence_list.sort(key=sentence_score(sentences, query, idfs))
    # return top n of list
    return sentence_list[-n:]


if __name__ == "__main__":
    main()
