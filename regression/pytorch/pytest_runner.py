"""
BRG tests on PyTorch => mainly used to to test HammerBlade device
Jan 31, 2020
Lin Cheng
"""
import sys
sys.argv = ["__main__.py"]
import pytest

pytest.main(["-vs", "/work/global/lc873/work/sdh/cosim/brg_bsg_bladerunner/bsg_f1/regression/pytorch/tests/test_hammerblade_device.py"])
