import os
import modal

app = modal.App("r2v-check-volume")
vol = modal.Volume.from_name("r2v-model-cache", create_if_missing=False)

VOL_MOUNT = "/model_cache"
image = modal.Image.debian_slim(python_version="3.10")


def find_hub(base: str) -> str:
    # Search base, base/*, base/*/* for a "hub" directory
    candidates = [base]
    for _ in range(3):
        new = []
        for c in candidates:
            hub = os.path.join(c, "hub")
            if os.path.isdir(hub):
                return hub
            try:
                for name in os.listdir(c):
                    p = os.path.join(c, name)
                    if os.path.isdir(p):
                        new.append(p)
            except Exception:
                pass
        candidates = new
    raise RuntimeError(f"Could not find 'hub' under {base} (searched 3 levels deep)")


@app.function(volumes={VOL_MOUNT: vol}, image=image)
def check():
    print("VOL_MOUNT =", VOL_MOUNT)
    print("VOL_MOUNT exists?", os.path.exists(VOL_MOUNT))
    print("VOL_MOUNT list:", os.listdir(VOL_MOUNT)[:50])

    hub_path = find_hub(VOL_MOUNT)
    hf_root = os.path.dirname(hub_path)

    print("DETECTED HF_ROOT =", hf_root)
    print("DETECTED HUB_PATH =", hub_path)

    hub_items = os.listdir(hub_path)
    print("HUB items count:", len(hub_items))
    print("HUB items sample:", hub_items[:30])

    model_dirs = [d for d in hub_items if d.startswith("models--")]
    print("Model dirs sample:", model_dirs[:10])


@app.local_entrypoint()
def main():
    check.remote()
