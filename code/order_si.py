# -*- coding: utf-8 -*-
"""Order the supplement by first citation in the main text.

The sections keep their stable keys, so nothing in either document has to be
renumbered by hand: the SI prints its numbers from a counter and the main text
cites the keys.  This script moves the blocks and rewrites the \\supplementary{}
list in the main text from the same order, so the two cannot disagree.

Usage:  python code/order_si.py [--check]
"""
import io
import os
import re
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
EN = os.path.join(ROOT, "submission", "paper_en")
SI = os.path.join(EN, "SI_main.tex")
MAIN = os.path.join(EN, "main.tex")


def read(p):
    with io.open(p, encoding="utf-8") as fh:
        return fh.read()


def write(p, s):
    with io.open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(s)


def strip_comments(src):
    out = []
    for line in src.split("\n"):
        keep, k = line, 0
        while True:
            j = line.find("%", k)
            if j < 0:
                break
            b, p = 0, j - 1
            while p >= 0 and line[p] == "\\":
                b += 1
                p -= 1
            if b % 2 == 1:
                k = j + 1
                continue
            keep = line[:j]
            break
        out.append(keep)
    return "\n".join(out)


def split_sections(si):
    head = list(re.finditer(r"\\sisection\{(.*?)\}\{([^}]+)\}", si))
    if not head:
        sys.exit("no \\sisection blocks found")
    out = []
    for i, m in enumerate(head):
        end = head[i + 1].start() if i + 1 < len(head) else len(si)
        out.append({"title": m.group(1).strip(), "key": m.group(2),
                    "text": si[m.start():end]})
    return si[:head[0].start()], out, si[head[-1].start():]


def first_citation_order(head, blocks, main_src):
    """Index of each block, ordered by its first citation in the main text."""
    key2i = {b["key"]: i for i, b in enumerate(blocks)}
    body = strip_comments(main_src)
    ev = []
    for m in re.finditer(r"\\siSec\{([^}]+)\}", body):
        if m.group(1) in key2i:
            ev.append((m.start(), key2i[m.group(1)]))
    for m in re.finditer(r"\\siSec\{([^}]+)\}\s*--\s*\\siSec\{([^}]+)\}", body):
        a, b = key2i.get(m.group(1)), key2i.get(m.group(2))
        if a is None or b is None:
            continue
        for i in range(min(a, b), max(a, b) + 1):
            ev.append((m.start(), i))
    # a float citation also sends the reader to the section that carries it.
    # The labels of the generated tables live in the \input files, not in the
    # block text, so the block has to be inlined before it is searched.
    for b in blocks:
        body_b = b["text"]
        for m in reversed(list(re.finditer(r"\\input\{([^}]+)\}", body_b))):
            name = m.group(1).strip()
            if not name.endswith(".tex"):
                name += ".tex"
            p = os.path.join(EN, name)
            if os.path.exists(p):
                body_b = body_b[:m.start()] + read(p) + body_b[m.end():]
        b["labels"] = set(re.findall(r"\\label\{([^}]+)\}", body_b))
    for mac in ("siTab", "siFig"):
        for m in re.finditer(r"\\%s\{([^}]+)\}" % mac, body):
            lab = m.group(1)
            for i, b in enumerate(blocks):
                if lab in b.get("labels", ()):
                    ev.append((m.start(), i))
    ev.sort()
    seen, order = set(), []
    for _, i in ev:
        if i not in seen:
            seen.add(i)
            order.append(i)
    missing = [blocks[i]["key"] for i in range(len(blocks)) if i not in seen]
    if missing:
        sys.exit("never cited, cannot order: %s" % missing)
    return order


def main():
    check = "--check" in sys.argv
    si = read(SI)
    main_src = read(MAIN)
    head, blocks, _ = split_sections(si)
    order = first_citation_order(head, blocks, main_src)
    target = [blocks[i]["key"] for i in order]
    now = [b["key"] for b in blocks]
    print("current order: %s" % ", ".join(k.split(":", 1)[1] for k in now))
    print("wanted  order: %s" % ", ".join(k.split(":", 1)[1] for k in target))
    if now == target:
        print("supplement already in citation order")
    elif check:
        print("FAIL: supplement is not in citation order")
        return 1
    else:
        new_si = head + "".join(blocks[i]["text"] for i in order)
        write(SI, new_si)
        print("rewrote SI_main.tex in citation order")

    # the \supplementary{} list is generated from the same order, not typed
    ordered = [blocks[i] for i in order]

    def sentence_case(t):
        """Keep the authored capitalisation.

        Lower-casing the first letter turned "Marmousi provenance" into
        "marmousi provenance" and "Bi-CGSTAB" into "bi-cgstab"; the list is
        generated, so a casing guess is a silent corruption of a name.  A
        trailing full stop is dropped because the entries are separated by
        semicolons.
        """
        return t.rstrip(".").strip()

    items = "; ".join("S%d, %s" % (k + 1, sentence_case(b["title"]))
                      for k, b in enumerate(ordered))
    start = main_src.index("\\supplementary{")
    end = main_src.index("\n", start)
    line = main_src[start:end]
    new_line = ("\\supplementary{The following supporting information can be "
                "downloaded at the article's web page, with the sections in the "
                "order the main text cites them: %s. The archive named in the "
                "Data Availability Statement is separate from this PDF and "
                "carries the code and raw records behind every table.}" % items)
    if line != new_line:
        if check:
            print("FAIL: the \\supplementary{} list does not match the SI order")
            return 1
        write(MAIN, main_src.replace(line, new_line))
        print("rewrote the \\supplementary{} list")
    return 0


if __name__ == "__main__":
    sys.exit(main())
