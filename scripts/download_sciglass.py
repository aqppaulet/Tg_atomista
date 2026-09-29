"""Download and verify the public SciGlass-derived GlassPy data snapshot."""

from pathlib import Path

from tellurite_tg.data import ensure_raw_data, file_md5


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[1]
    properties, compositions = ensure_raw_data(root / "data" / "raw")
    archive = root / "data" / "raw" / "GlassPy_files.zip"
    print(f"Archive: {archive}")
    print(f"MD5: {file_md5(archive)}")
    print(f"Properties: {properties}")
    print(f"Compositions: {compositions}")
