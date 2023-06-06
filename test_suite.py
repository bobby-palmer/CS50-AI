from parser_1 import *
import os
import pytest
TEXT_PATH = "sentences"

@pytest.mark.parametrize("file_num",range(1,11))
def test_can_parse(file_num):
        with open(os.path.join(TEXT_PATH, str(file_num)+".txt")) as f:
            s = f.read()
            s = preprocess(s)
        trees = list(parser.parse(s))
        assert len(trees) != 0
    