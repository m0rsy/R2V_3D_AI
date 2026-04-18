import modal

app = modal.App("r2v-clear-jobs")
jobs = modal.Dict.from_name("r2v-jobs", create_if_missing=False)

@app.local_entrypoint()
def main():
    keys = list(jobs.keys())
    print(f"Found {len(keys)} jobs")
    for k in keys:
        del jobs[k]
    print("✅ All jobs cleared")
