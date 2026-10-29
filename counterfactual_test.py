"""Compatibility entry point; the implementation lives in lob_xai."""
import sys
from lob_xai import counterfactual_test as _implementation

if __name__ == "__main__":
    _implementation.main()
else:
    # Preserve existing imports, including private helpers, without a second copy.
    sys.modules[__name__] = _implementation
