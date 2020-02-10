import pytest

"""
BRG tests on PyTorch => tests of real offloading kernels
Feb 09, 2020
Lin Cheng
"""

# test of adding two tensors
def test_elementwise_add_1():
  import torch
  x1 = torch.rand(1,10)
  x2 = torch.rand(1,10)
  y = x1 + x2
  y_h = x1.hammerblade() + x2.hammerblade()
  assert y_h.device == torch.device("hammerblade")
  assert torch.equal(y_h.cpu(), y)

