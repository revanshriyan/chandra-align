import os
import sys
import pytest

def test_environment_variables_and_imports():
    """Verify system requirements and imports necessary for containerized execution."""
    # Ensure critical dependencies load without throwing library link errors
    import cv2
    import numpy as np
    from chandra_align.matching.deep_matchers import DeepMatcherChain
    from chandra_align.preprocessing.sar import refined_lee_filter

    assert cv2.__version__ is not None
    assert np.__version__ is not None

    # Check non-root environment or standard path structures
    chain = DeepMatcherChain()
    assert chain is not None

def test_headless_opencv_backend():
    """Ensure OpenCV operates in headless mode without GUI display errors."""
    import cv2
    import numpy as np
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    assert gray.shape == (100, 100)