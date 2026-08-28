"""tests/run_all.py
===================
Master test runner.  Run from the project root:

    python tests/run_all.py

Exits with code 0 on all-pass, 1 on any failure.
"""
import sys, os, unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

loader  = unittest.TestLoader()
suite   = loader.discover(start_dir=os.path.dirname(__file__), pattern="test_*.py")
runner  = unittest.TextTestRunner(verbosity=2)
result  = runner.run(suite)
sys.exit(0 if result.wasSuccessful() else 1)
