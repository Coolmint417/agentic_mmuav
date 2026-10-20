"""One-time download of the CUDA kernel used by Transformers 5.18."""
import os

# Some desktop proxy setups export socks://, unsupported by HTTPX.
for name in ("ALL_PROXY", "all_proxy"):
    if os.environ.get(name, "").startswith("socks://"):
        os.environ.pop(name)
os.environ.pop("HF_HUB_OFFLINE", None)

from kernels import get_kernel
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    # Pin the v4 commit used by agent/ministral.py, avoiding legacy ref collisions.
    # Hub 1.33 checks snapshot completeness in offline mode. Cache the whole
    # small source repository, including the other backend variants and README.
    snapshot_download("kernels-community/finegrained-fp8", repo_type="kernel",
                      revision="8c178950e8710e26a2210e4e909cd02dc16c8715")
    kernel = get_kernel("kernels-community/finegrained-fp8",
                        revision="8c178950e8710e26a2210e4e909cd02dc16c8715")
    print("FP8 v4 kernel ready:", kernel.__file__)
