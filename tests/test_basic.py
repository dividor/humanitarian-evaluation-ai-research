#!/usr/bin/env python3
"""
test_basic.py - Basic tests for pipeline components
"""
import sys
import tempfile


def test_parse_module():
    """Test that parse module can be imported and basic functions work"""
    print("Testing parse module...")

    try:
        from parse import PDFParser

        # Create a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            parser = PDFParser(pdf_dir=tmpdir, output_dir=tmpdir)

            # Test is_heading method
            assert parser.is_heading("INTRODUCTION") is True
            assert parser.is_heading("1. First Section") is True
            assert parser.is_heading("This is regular text.") is False

            # Test estimate_heading_level
            level = parser.estimate_heading_level("1. First Section")
            assert level == 1

            print("✓ Parse module tests passed")
            return True
    except Exception as e:
        print(f"✗ Parse module tests failed: {e}")
        return False


def test_index_module():
    """Test that index module can be imported"""
    print("Testing index module...")

    try:
        from index import VectorIndexer

        # Just test that we can instantiate the class
        # (won't actually connect to DB)
        indexer = VectorIndexer(parsed_dir="./parsed")

        # Check that model loads (this will download on first run)
        print(f"  - Embedding dimension: {indexer.embedding_dim}")

        print("✓ Index module tests passed")
        return True
    except Exception as e:
        print(f"✗ Index module tests failed: {e}")
        return False


def test_app_module():
    """Test that app module can be imported"""
    print("Testing app module...")

    try:
        # We can't fully test Flask app without DB, but we can check imports
        import app

        assert hasattr(app, "app")
        assert hasattr(app, "search")

        print("✓ App module tests passed")
        return True
    except Exception as e:
        print(f"✗ App module tests failed: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("Running basic component tests")
    print("=" * 60)

    results = []

    # Test parse module
    results.append(("Parse", test_parse_module()))

    # Test index module (requires downloading model)
    results.append(("Index", test_index_module()))

    # Test app module
    results.append(("App", test_app_module()))

    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"{name}: {status}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
