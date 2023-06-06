from questions import load_files, tokenize, get_all_words, num_docs_with_word

TEST_FILES = "test_files"

def test_load_file():
    my_dict = load_files(TEST_FILES)
    assert len(my_dict) == 1

def test_load_file_dictionary_name():
    my_dict = load_files(TEST_FILES)
    assert "test_document.txt" in my_dict

def test_load_file_dictionary_values():
    my_dict = load_files(TEST_FILES)
    assert "test document content" in my_dict.values()

def test_tokenize():
    document = "his, Elephant!"
    tokens = tokenize(document)
    assert tokens == ["elephant"]


def test_get_all_words():
    documents = {
        'doc1':"doc content one",
        'doc2':"doc content two"
    }
    words = get_all_words(documents)
    assert words == {"doc", "content", "one", "two"}


def test_get_num_words():
    documents = {
        'doc1':"doc content one",
        'doc2':"doc content two"
    }
    assert num_docs_with_word("doc", documents) == 2
    assert num_docs_with_word("one", documents) == 1
    assert num_docs_with_word("fail", documents) == 0
