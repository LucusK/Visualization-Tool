#assumes you already ran docker compose and Weaviate is reachable at http://localhost:8080

from pathlib import Path
import numpy as np
import weaviate
from weaviate.classes.config import Configure, Property, DataType

WEAVIATE_HTTP = "http://localhost:8080"
CLASS_NAME = "MuveraDoc"

BASE_DIR = Path(__file__).resolve().parents[1]
FDE_DIR = BASE_DIR / "muvera_out" / "fde"

def main():
    if not FDE_DIR.exists():
        raise RuntimeError(f"Missing FDE dir: {FDE_DIR}")

    # v4 client (recommended)
    client = weaviate.connect_to_local(host="localhost", port=8080, skip_init_checks=True)

    try:
        # Create collection if it doesn't exist (bring-your-own vectors)
        if not client.collections.exists(CLASS_NAME):
            client.collections.create(
                name=CLASS_NAME,
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="doc_id", data_type=DataType.TEXT),
                    Property(name="fde_file", data_type=DataType.TEXT),
                ],
            )

        col = client.collections.get(CLASS_NAME)

        npy_files = sorted(FDE_DIR.glob("*.npy"))
        if not npy_files:
            raise RuntimeError(f"No .npy files found in: {FDE_DIR}")

        inserted = 0
        for f in npy_files:
            v = np.load(f).astype(np.float32)

            # Convert (1, D) -> (D,)
            if v.ndim == 2 and v.shape[0] == 1:
                v = v[0]

            if v.ndim != 1:
                raise RuntimeError(f"Unexpected FDE shape in {f.name}: {v.shape}")

            doc_id = f.stem  # e.g. sample_FDE

            col.data.insert(
                properties={
                    "doc_id": doc_id,
                    "fde_file": str(f),
                },
                vector=v.tolist(),
            )
            inserted += 1

        print(f"Inserted {inserted} docs into Weaviate collection '{CLASS_NAME}'")

    finally:
        client.close()

if __name__ == "__main__":
    main()