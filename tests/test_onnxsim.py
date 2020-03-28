import urllib.request
import tempfile

import onnxsim
import onnx


def simplify_url(url: str, **kwargs) -> bool:
    with tempfile.NamedTemporaryFile() as f:
        response = urllib.request.urlopen(url)
        f.write(response.read())
        _, check_ok = onnxsim.simplify(f.name, **kwargs)
        return check_ok
    

def test_mobileseg():
    assert simplify_url("https://drive.google.com/uc?export=download&id=1EEencrovhayXTuqVL-zzV5eD_NoBEpfm", perform_optimization=False, input_shapes={'input_1': [1, 224, 224, 3]})


def test_pfe():
    assert simplify_url("https://drive.google.com/uc?export=download&id=1ZML6j-88jR6DfRNBtYZZ-CHR7fttU9JR")


def test_pytorch13():
    assert simplify_url("https://drive.google.com/uc?export=download&id=1DJ4Fqa5Lsku4Zuu96ax4r7BO0unlYshv")


def test_trk():
    assert simplify_url("https://drive.google.com/uc?export=download&id=1S66i5NLFszuIJRvR9tENWkB1hJ7T-bh8")


def test_bisenet():
    assert simplify_url("https://drive.google.com/uc?export=download&id=1QbJ_S6ob6lVvIy3qFLHPelHlg5BN_3-3")
