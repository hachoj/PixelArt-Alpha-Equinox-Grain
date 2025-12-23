import os
import sys
import inspect
from pathlib import Path

# Boilerplate from train_stage2.py
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"

conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    ca_bundle = Path(conda_prefix) / "ssl" / "cacert.pem"
    if ca_bundle.exists():
        os.environ.setdefault("SSL_CERT_FILE", str(ca_bundle))
        os.environ.setdefault("CURL_CA_BUNDLE", str(ca_bundle))
        os.environ.setdefault("REQUESTS_CA_BUNDLE", str(ca_bundle))
        print("Using CA bundle:", ca_bundle)
    else:
        print("Conda CA bundle not found at:", ca_bundle)
else:
    print("CONDA_PREFIX not set; leaving SSL cert settings unchanged.")

try:
    import gemma
    from gemma import gm
    import gemma.research.t5gemma.modules as t5gemma_modules

    if hasattr(t5gemma_modules, "Attention"):
        att_class = t5gemma_modules.Attention
        print(f"\nSource code for {att_class}:")
        print(inspect.getsource(att_class))
    else:
        print("Attention class not found in modules.")

except Exception as e:
    print(f"An error occurred: {e}")
    import traceback

    traceback.print_exc()
