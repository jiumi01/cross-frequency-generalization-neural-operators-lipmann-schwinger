# Author: Longgang Wang
"""Publish the local HEAD to GitHub through the REST API.

Why not `git push`: this machine reaches api.github.com (through `gh`, which
honours the session proxy) but not github.com over the git transport -- the
TLS handshake and the CONNECT tunnel both fail.  The repository is therefore
published tree-first: blobs for the files that differ, a tree based on the
remote head's tree, one commit, one ref update.

The result is byte-identical to a push of the same content: the tree the API
builds equals the local HEAD tree (the script asserts it), so the commit hash
differs only because the committer identity and timestamp do.

Usage
-----
    python code/publish_api.py --dry-run     # list what would change
    python code/publish_api.py              # publish, print the new commit
"""
import os
import sys
import json
import base64
import subprocess
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.environ.get(
    "LS_REPO", "jiumi01/cross-frequency-generalization-neural-operators-lipmann-schwinger")


def sh(*args):
    p = subprocess.run(args, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit("git %s failed: %s" % (" ".join(args), p.stderr.strip()))
    return p.stdout


def gh(method, path, body=None):
    cmd = ["gh", "api", "-X", method, path]
    if body is not None:
        cmd += ["--input", "-"]
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                       input=json.dumps(body) if body is not None else None)
    if p.returncode != 0:
        raise SystemExit("gh api %s %s failed: %s" % (method, path, p.stderr.strip()))
    return json.loads(p.stdout) if p.stdout.strip() else {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    ref = gh("GET", "repos/%s/git/ref/heads/main" % REPO)
    head = ref["object"]["sha"]
    head_tree = gh("GET", "repos/%s/git/commits/%s" % (REPO, head))["tree"]["sha"]
    local = sh("git", "rev-parse", "HEAD").strip()
    local_tree = sh("git", "rev-parse", "HEAD^{tree}").strip()
    print("remote head %s (tree %s)" % (head[:7], head_tree[:7]))
    print("local  head %s (tree %s)" % (local[:7], local_tree[:7]))

    if head == local:
        print("already at the local head; nothing to publish")
        return
    rc = subprocess.run(["git", "merge-base", "--is-ancestor", head, local],
                        cwd=ROOT).returncode
    if rc != 0:
        raise SystemExit("remote head is not an ancestor of local HEAD; refusing")

    diff = sh("git", "diff", "--name-status", "%s..%s" % (head, local)).strip().splitlines()
    changes = []
    for line in diff:
        parts = line.split("\t")
        st, path = parts[0], parts[-1]
        changes.append((st, path))
    print("%d changed paths" % len(changes))
    for st, path in changes:
        print("  %s  %s" % (st, path))
    if a.dry_run:
        return

    entries = []
    for st, path in changes:
        if st.startswith("D"):
            entries.append(dict(path=path, mode="100644", type="blob", sha=None))
            continue
        # Content is read from the object store, not from the working file.
        # With core.autocrlf=true the two differ (the index holds LF, a
        # checkout would hold CRLF), and uploading the disk bytes would build a
        # tree that is not HEAD's -- the assertion below would fire on a
        # publish that is otherwise correct, and the obvious "fix" (dropping
        # the assertion) would let the remote diverge from the audited state.
        blob_bytes = subprocess.run(["git", "cat-file", "blob", "HEAD:" + path],
                                    cwd=ROOT, capture_output=True, check=True).stdout
        content = base64.b64encode(blob_bytes).decode()
        blob = gh("POST", "repos/%s/git/blobs" % REPO,
                  dict(content=content, encoding="base64"))
        entries.append(dict(path=path, mode="100644", type="blob", sha=blob["sha"]))
    tree = gh("POST", "repos/%s/git/trees" % REPO,
              dict(base_tree=head_tree, tree=entries))
    if tree["sha"] != local_tree:
        raise SystemExit("built tree %s != local tree %s -- refusing to publish a "
                         "state that is not the checkout" % (tree["sha"][:7], local_tree[:7]))
    msg = sh("git", "log", "-1", "--pretty=%B", local).strip()
    commit = gh("POST", "repos/%s/git/commits" % REPO,
                dict(message=msg, tree=tree["sha"], parents=[head]))
    gh("PATCH", "repos/%s/git/refs/heads/main" % REPO, dict(sha=commit["sha"], force=False))
    print("published %s (tree %s)" % (commit["sha"], commit["sha"][:7]))
    print("set DAS commit to %s, tree to %s" % (commit["sha"][:7], tree["sha"][:7]))


if __name__ == "__main__":
    main()
