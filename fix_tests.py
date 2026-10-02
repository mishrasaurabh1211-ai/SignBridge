from pathlib import Path

p = Path("tests/test_data.py")
content = p.read_text(encoding="utf-8")

old_code = """    # 250 labels map
    sign_map = {f"sign_{i}": i for i in range(250)}
    sign_map["thankyou"] = 0
    sign_map["tv"] = 1
    # ensure total keys is exactly 250
    final_sign_map = {f"sign_{i}": i for i in range(250)}
    final_sign_map["thankyou"] = 0
    final_sign_map["tv"] = 1
    if len(final_sign_map) > 250:
        final_sign_map = dict(list(final_sign_map.items())[:250])"""

new_code = """    # 250 labels map ensuring 'thankyou' and 'tv' are kept
    final_sign_map = {"thankyou": 0, "tv": 1}
    for i in range(248):
        final_sign_map[f"sign_{i}"] = i + 2"""

content = content.replace(old_code, new_code)
p.write_text(content, encoding="utf-8")
print("Test fixture patched successfully.")