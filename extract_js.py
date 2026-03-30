import re
import traceback

with open(r"c:\Users\jihyeon\Desktop\my\static\index.html", "r", encoding="utf-8") as f:
    html = f.read()

script_match = re.search(r"<script>(.*?)</script>", html, re.DOTALL)
if script_match:
    js_code = script_match.group(1)
    with open("tmp_script.js", "w", encoding="utf-8") as f:
        f.write(js_code)
    print("Extracted script to tmp_script.js")
else:
    print("No script tag found!")
