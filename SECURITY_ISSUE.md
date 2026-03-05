# Security Issue: SSRF and Input Validation in External Fetch Utilities

## Overview

Two public helper functions in `pyaptamer` previously allowed uncontrolled
network access based on user input:

1. `pyaptamer.datasets._loaders._hf_to_dataset_loader.load_hf_to_dataset`
   (when `download_locally=True`), which downloaded *any* HTTP URL provided by
   the caller.
2. `pyaptamer.utils._pdb_to_seq_uniprot.pdb_to_seq_uniprot` which interpolated
   a user-supplied PDB identifier directly into two API endpoints.

In a scenario where these utilities are used with untrusted data (e.g. as part
of a web service, shared notebook, or processing pipeline), an attacker could
provide a malicious value leading to **Server-Side Request Forgery (SSRF)**,
remote file downloads, or interaction with internal network resources.

## Impact

- Remote attackers or untrusted users can coerce the application into
  contacting arbitrary hosts, including internal IP addresses (e.g.
  `http://169.254.169.254` or `http://localhost:8000`).
- Large downloads from external hosts may exhaust network bandwidth or disk
  space, effectively causing a denial of service.
- Writing arbitrary content under `./hf_datasets` (using a crafted filename) is
  possible, potentially overwriting important files.
- The PDB helper could generate malformed URLs or query unexpected APIs if the
  `pdb_id` parameter contains shell characters or path traversal.

This affects any deployment of the package that processes external metadata or
URLs, which is common given its use-case in bioinformatics workflows.

## Proof of Concept

The following script demonstrates how the vulnerable loader could be abused.
Run it **with the pre-patch version** of the library to see the unsafe
behaviour; after applying the fixes included in this branch it will raise an
error instead.

```python
# examples/ssrf_poc.py
from pyaptamer.datasets import load_hf_to_dataset

# attacker-controlled URL pointing to an internal service or arbitrary host
malicious_url = "http://example.com/sensitive.txt"

print("attempting to download", malicious_url)
try:
    # prior to the patch this would download the file and save it in ./hf_datasets
    load_hf_to_dataset(malicious_url, download_locally=True)
    print("download succeeded (vulnerable)")
except Exception as err:
    print("blocked or error:", err)
```

Similarly, the PDB utility can be abused:

```python
from pyaptamer.utils import pdb_to_seq_uniprot

# malformed pdb id leads to request URL injection
print(pdb_to_seq_uniprot("../etc/passwd"))
```

## Fix and Mitigation

- Added strict regex validation for `pdb_id` inputs.
- Implemented `_validate_hf_url` to restrict downloads to the `huggingface.co`
  domain and applied it in all relevant code paths.
- Added comprehensive unit tests covering both positive and negative cases.

These changes are included in branch `enh/sklearn-delegator` along with the
original sklearn-delegator enhancement. The fix is backwards-compatible and
will not break existing, non-adversarial use.

## Code Changes Summary

For reference, the following key modifications implement the fix:

```diff
+# pyaptamer/utils/_pdb_to_seq_uniprot.py
+ import re
+
+ if not re.fullmatch(r"[0-9][A-Za-z0-9]{3}", pdb_id):
+     raise ValueError(f"Invalid PDB ID '{pdb_id}'")
```

```diff
+# pyaptamer/datasets/_loaders/_hf_to_dataset_loader.py
+_ALLOWED_HF_HOSTS = ("huggingface.co", "hf.co")
+
+def _validate_hf_url(url: str) -> None:
+    parsed = urllib.parse.urlparse(url)
+    if parsed.scheme not in ("http", "https"):
+        raise ValueError(f"URL scheme {parsed.scheme!r} not allowed")
+    host = parsed.hostname or ""
+    if not any(host.endswith(ah) for ah in _ALLOWED_HF_HOSTS):
+        raise ValueError(f"Host {host!r} not permitted for download")
+
+# applied validation calls in `_download_to_cwd` and loader entrypoint
```

(see branch `enh/sklearn-delegator` for full diffs).

## Test Coverage

New tests added ensure the vulnerabilities are caught:

* `pyaptamer/utils/tests/test_pdb_to_seq_uniprot.py`
  - valid fetch, invalid-ID and nonexistent-ID cases.
* `pyaptamer/datasets/tests/test_hf_to_dataset.py`
  - valid load, disallowed-host rejection, suggested allowed-host path.
* existing delegation tests in `pyaptamer/utils/tests/test_sklearn_delegator.py` remain unaffected.

All tests run successfully: `337 passed, 1 skipped` on our CI.

## Additional Proofs

- **Example PoC script** at `examples/ssrf_poc.py` demonstrates exploitation
  and blocked behaviour post-patch.
- Manual testing of `pdb_to_seq_uniprot` with invalid IDs verifies the check.

## Recommended Actions

1. Merge the branch and release a new patch version.
2. Raise awareness among users (e.g. via CHANGELOG or security advisory).
3. Consider adding a dedicated security section to documentation.
4. Monitor related endpoints (HF loader, PDBe API) for abnormal use.

---

This issue is **high priority** due to the clear exploitability in real-world
contexts and the sensitivity of biological data processed by the library.

Assign to: @kallal79 (primary fixer) and notify the security team.

---

This issue is **high priority** due to the clear exploitability in real-world
contexts and the sensitivity of biological data processed by the library.

Assign to: @kallal79 (primary fixer) and notify the security team.