# scripts/inspect_dhsnn.py

from pathlib import Path
import pkgutil
import importlib

from models.dh_snn_wrapper import add_dhsnn_to_path

def main():
    dhsnn_path = add_dhsnn_to_path()
    print(f"[OK] DH-SNN path: {dhsnn_path}")

    # Try importing the main folder mentioned in README
    try:
        import SNN_layers  # noqa: F401
        print("[OK] import SNN_layers succeeded")
    except Exception as e:
        print("[FAIL] import SNN_layers failed:", e)
        raise

    # List available submodules under SNN_layers
    print("\n[SNN_layers submodules]")
    import SNN_layers
    for m in pkgutil.iter_modules(SNN_layers.__path__):
        print(" -", m.name)

if __name__ == "__main__":
    main()
