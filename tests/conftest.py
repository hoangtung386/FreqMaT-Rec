import pytest
import torch

CUDA = torch.cuda.is_available()


def pytest_collection_modifyitems(items):
    """Skip tests marked ``@pytest.mark.cuda`` when no GPU is present."""
    skip = pytest.mark.skip(reason="needs CUDA (Mamba-3 Triton kernels)")
    for item in items:
        if "cuda" in item.keywords and not CUDA:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def device() -> str:
    return "cuda" if CUDA else "cpu"


@pytest.fixture
def x(device):
    torch.manual_seed(0)
    return torch.randn(4, 50, 64, device=device, requires_grad=True)
