"""Check GitHub for a newer version of the app and install it.  No account or git needed.

Which version you have: the folder's git commit if it is a git clone, else version.txt (written by the last update here).
An update replaces only the program files (*.py, *.bat, *.ps1, the icon, README, tools\\, docs\\). Your settings, voices,
transcripts, models and the .venv are never touched. If install.ps1 changed, run install.bat again afterwards."""
import os, io, json, shutil, subprocess, tempfile, urllib.request, zipfile

REPO = "Navotj/Subtitles"
API = f"https://api.github.com/repos/{REPO}"
HERE = os.path.dirname(os.path.abspath(__file__))
CODE_EXT = (".py", ".bat", ".ps1", ".ico", ".md")
CODE_DIRS = ("tools", "docs")
KEEP = {"config.json", "my_voice.json", "app.log", "install_log.txt"}      # never overwritten, even if they appear in the repo


def _get(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "subtitles-updater", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def is_clone(root=HERE):
    return os.path.isdir(os.path.join(root, ".git"))


def _git(root, *args):
    r = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=120,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return r.returncode, (r.stdout + r.stderr).strip()


def local_version(root=HERE):
    """full commit hash of what is installed, or None when unknown"""
    if is_clone(root):
        try:
            rc, out = _git(root, "rev-parse", "HEAD")
            if rc == 0 and len(out) >= 40: return out[:40]
        except Exception: pass
    try:
        return open(os.path.join(root, "version.txt"), encoding="utf-8").read().strip() or None
    except OSError:
        return None


def latest():
    """-> {"sha", "message", "date", "branch"} of the newest commit on the repository's default branch"""
    branch = json.loads(_get(API))["default_branch"]
    c = json.loads(_get(f"{API}/commits/{branch}"))
    return {"sha": c["sha"], "branch": branch, "message": c["commit"]["message"].strip().splitlines()[0],
            "date": c["commit"]["committer"]["date"][:10]}


def behind_by(local, remote):
    """how many commits the installed version is behind (None if it cannot be told)"""
    if not local: return None
    try:
        d = json.loads(_get(f"{API}/compare/{local}...{remote}"))
        return int(d["ahead_by"]) if d.get("status") != "identical" else 0
    except Exception:
        return None


def check(root=HERE):
    """-> dict(state = "current" | "update" | "unknown", local, latest, behind)"""
    new = latest(); loc = local_version(root)
    if loc == new["sha"]: return dict(state="current", local=loc, latest=new, behind=0)
    if loc is None: return dict(state="unknown", local=None, latest=new, behind=None)
    n = behind_by(loc, new["sha"])
    if n == 0: return dict(state="current", local=loc, latest=new, behind=0)       # you are ahead of the published version
    return dict(state="update", local=loc, latest=new, behind=n)


def _wanted(rel):
    rel = rel.replace("\\", "/")
    if os.path.basename(rel) in KEEP: return False
    if "/" in rel: return rel.split("/")[0] in CODE_DIRS
    return rel.lower().endswith(CODE_EXT)


def install_zip(sha, root=HERE, data=None):
    """download the repository at sha and copy the program files over the installed ones -> list of changed files"""
    if data is None: data = _get(f"https://github.com/{REPO}/archive/{sha}.zip", timeout=120)
    changed = []
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        for info in z.infolist():
            if info.is_dir(): continue
            rel = info.filename.split("/", 1)[1] if "/" in info.filename else ""
            if not rel or not _wanted(rel): continue
            new = z.read(info)
            dst = os.path.join(root, *rel.split("/"))
            if os.path.exists(dst) and open(dst, "rb").read() == new: continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            tmp = dst + ".update"
            with open(tmp, "wb") as f: f.write(new)
            os.replace(tmp, dst); changed.append(rel)
    open(os.path.join(root, "version.txt"), "w", encoding="utf-8").write(sha + "\n")
    return changed


def install(sha, root=HERE):
    """-> (ok, text, changed files). A git clone is pulled (fast-forward only, so your own edits are never overwritten)."""
    if is_clone(root):
        try:
            rc, out = _git(root, "status", "--porcelain", "--untracked-files=no")
            if out: return False, "This folder is a git clone with changes of your own. Commit or stash them, then run: git pull", []
            before = local_version(root)
            rc, out = _git(root, "pull", "--ff-only")
            if rc != 0: return False, "git pull failed:\n" + out, []
            rc, files = _git(root, "diff", "--name-only", before or "HEAD", "HEAD")
            return True, out, [f for f in files.splitlines() if f]
        except FileNotFoundError:
            pass                                                      # no git installed: fall back to the zip
    try:
        return True, "", install_zip(sha, root)
    except Exception as e:
        return False, f"download failed: {e}", []
