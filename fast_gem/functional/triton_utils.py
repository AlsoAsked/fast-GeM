import triton
import triton.language as tl


@triton.jit
def pow(x, p):
    return tl.exp(tl.log(x) * p)
