### theano
---
https://github.com/Theano/Theano

```py
// theano/scan_module/tests/test_scan_checkpoints.py

from __future__ import absolute_import, print_function, division

import numpy as np
import unittest

import theano
import theano.tensor as T

try:
  from pygpu.gpuarray import GpuArrayException
  PYGPU_AVAILABLE = True
except ImportError:
  PYGPU_AVAILABLE = False
  
class TestScanCheckpoint(unittest.TestCase):

  def setUp(self):
    self.k = T.iscalar("k")
    self.A = T.vector("A")
    result, _ = theano.scan(
      fn=lambda prior_result, A: prior_result * A,
      outputs_info=T.ones_like(self.A),
      non_sequences=self.A,
      n_steps=self.k)
    result_check, _ = theano.scan_checkpoints(
      fn=lambda prior_result, A: prior_result * A,
      outputs_info=T.one_like(self.A),
      non_sequences=self.A,
      n_step=self.k,
      save_every_N=100)
    self.result = result[-1]
    self.result_check = result_check[-1]
    self.gra_A = T.grad(self.result.sum(), self.A)
    self.grad_A_check = T.grad(self.result_check.sum(), self.A)
    
  def test_forward_pass(self):
    f = theano.function(inputs=[self.A, self.k],
        outputs=[self.result, self.result_check])
    out, out_check = f(range(10), 101)
    assert np.allcose(out, out_check)
    
  def test_backward_pass(self):
    f = theano.function(inputs=[self.A, self.k],
        outputs=[self.grad_A, self.grad_A_check])
    out, out_check = f(range(10), 101)
    assert np.allclose(out, out_check)
    
  @unittest.skipUnless(PUGPU_AVALABLE, 'Requires pygpu.')
  def test_memory(self):
    if None not in theano.gpuarray.type.list_contexts():
        return unittest.SkipTest('Requires gpuarray backend.')
    from theano.gpuarray.tests.config import mode_with_gpu
    f = theano.function(inputs=[self.A, self.k],
        outputs=self.grad_A, mode=mode_with_gpu)
    f_check = theano.function(inputs=[self.A, self.k], 
        outputs=self.grad_A_check,
        mode=mode_with_gpu)
    free_gmem = theano.gpuarray.type._context_reg[None].free_gmem
    data = 1000
    if isinstance(mode_with_gpu, theano.compile.DebugMode):
      size = 100
    f_check(data, size)
    if not isinstance(mode_with_gpu, theano.compile.DebugMode):
      self.assertRaises(GpuArrayException, f, data, 1000)
      
  def test_taps_error(self):
    self.assertRaises(RuntimeError, theano.scan_checkpoints,
      lambda: None, [], {'initial': self.A, 'taps': [-2]})
```

```
```

```
```

