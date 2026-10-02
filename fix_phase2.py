from pathlib import Path

# 1. Fix pyproject.toml mypy overrides
toml_path = Path("pyproject.toml")
toml_content = toml_path.read_text(encoding="utf-8")
toml_content = toml_content.replace(
    '"matplotlib.*"', 
    '"matplotlib.*",\n    "pandas.*",\n    "tqdm.*"'
)
toml_path.write_text(toml_content, encoding="utf-8")

# 2. Fix cache.py PERF402
cache_path = Path("signbridge/data/cache.py")
cache_content = cache_path.read_text(encoding="utf-8")

old_loop = """            for res in tqdm(
                pool.imap_unordered(_process_single_sequence, tasks, chunksize=chunksize),
                total=len(tasks),
                desc="Caching sequences",
            ):
                results.append(res)"""

new_loop = """            results.extend(tqdm(
                pool.imap_unordered(_process_single_sequence, tasks, chunksize=chunksize),
                total=len(tasks),
                desc="Caching sequences",
            ))"""

cache_content = cache_content.replace(old_loop, new_loop)
cache_path.write_text(cache_content, encoding="utf-8")

print("Patches applied successfully.")