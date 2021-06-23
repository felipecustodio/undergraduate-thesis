import os

os.system(
    "pandoc \
    --pdf-engine=xelatex \
    report.md -o report.pdf"
)
