import pytest
import torch
from memorytest import MemoryTest, create_memorytest

class TestMemoryTest:
    def test_init(self):
        m = MemoryTest(dim=256, hidden_dim=512)
        assert m.dim == 256
    
    def test_forward(self):
        m = MemoryTest(dim=256)
        x = torch.randn(32, 256)
        assert m(x).shape == (32, 256)
    
    def test_factory(self):
        m = create_memorytest()
        assert isinstance(m, MemoryTest)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])
