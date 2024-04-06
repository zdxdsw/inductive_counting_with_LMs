import subprocess

filesize_limit = "8M"
dirs = ["scripts/"]

result = ""
for d in dirs:
    command = f"find {d} -size +{filesize_limit} -print0 | du -h --files0-from=- | sort -h"
    result += subprocess.check_output(command, shell=True, text=True)


with open(".gitignore", "r") as f:
    lines = f.readlines()
    top_lines = []
    for line in lines:
        top_lines.append(line)
        if "Auto discovered large files" in line: break
with open(".gitignore", "w") as f:
    f.write("".join(top_lines))
    for r in result.strip().split("\n"):
        f.write(r.split("\t")[1] + "\n")
