---
name: log-commit
description: Stage changed files, write an informative commit message, update COMMIT-HISTORY.md, and tell the user to push themselves. Use when the user says "commit", "save my changes", or asks to commit.
allowed-tools: Bash, Read, Edit, Write
---

# /log-commit Skill

When this skill is invoked, follow these steps **in order**:

## Step 0 — Resolve Target Repository

Check if the user passed a directory argument (e.g. `/log-commit weaviate` or `/log-commit Embedding-Visualization-Tool`).

**If an argument was given:** use that path as the working directory for all subsequent git commands. Verify it is a git repo:
```bash
git -C <path> status
```
If it fails, tell the user the path is not a git repository and stop.

**If no argument was given:** discover all git repos under the current working directory:
```bash
find . -maxdepth 2 -name ".git" -type d -not -path "*/.git/.git" | sed 's|/.git||' | sort
```
If exactly one repo is found, use it. If multiple are found, list them and ask:
> "Which repo should I commit to? (e.g. `weaviate`, `Embedding-Visualization-Tool`)"
Wait for the user's reply before continuing.

All git commands from Step 1 onward must use `git -C <resolved-path>` instead of plain `git`.

## Step 1 — Discover What Changed

Run:
```bash
git -C <resolved-path> status
git -C <resolved-path> diff --name-only HEAD
git -C <resolved-path> diff --stat HEAD
```

Review the output to understand which files changed and what kind of changes they are (new file, modified, deleted).

## Step 2 — Stage the Changes

Using the file list from Step 1, stage each relevant file **by name**:
```bash
git -C <resolved-path> add path/to/file1 path/to/file2 ...
```

Rules:
- Do **not** use `git add -A` or `git add .` — these can accidentally include secrets, credentials, or large binaries.
- Skip any file that looks like it could contain sensitive data (`.env`, `*credentials*`, `*secret*`, etc.) and warn the user.
- Respect `.gitignore` — do not force-add ignored files.

## Step 3 — Write a Commit Message

Write a short but informative commit message. Rules:
- First line: 50 chars or fewer, imperative mood ("Add X", "Fix Y", "Refactor Z")
- Do NOT use generic messages like "update files" or "misc changes"
- The message should describe *what* changed and *why* at a high level

Then commit:
```bash
git -C <resolved-path> commit -m "<your message here>"
```

## Step 4 — Get the Timestamp

Try each of these in order, stopping at the first one that produces output:
```bash
date "+%Y-%m-%d %H:%M %Z"
```
```bash
powershell -Command "Get-Date -Format 'yyyy-MM-dd HH:mm'"
```

If both fail or return no output, **ask the user**: "What is the current date and time? I need it for COMMIT-HISTORY.md."

Wait for their reply before continuing.

## Step 6 — Update COMMIT-HISTORY.md

Locate `COMMIT-HISTORY.md` by searching from the repo root:
```bash
find . -name "COMMIT-HISTORY.md" -not -path "*/.git/*" 2>/dev/null
```

If it does **not** exist, create it with this header:
```markdown
# Commit History

A running log of all commits with summaries of what changed and why.

---
```

Then **append** a new entry to the bottom of `COMMIT-HISTORY.md` using this format:

```markdown
## [YYYY-MM-DD HH:MM TZ] — <commit message here>

**Commit:** `<full git commit hash>`
**Branch:** `<branch name>`

### Changes
- `path/to/file.ext` — Brief plain-English explanation of what changed and why
- `another/file.ts` — Same: what and why, not just "modified"
- *(repeat for each changed file)*

---
```

Rules for the Changes list:
- One bullet per file
- Use backticks around the file path
- Write the explanation for a human who wasn't watching — what does this file do, and what specifically changed?
- If a file was **deleted**, note it: "`old-file.py` — Removed; replaced by `new-file.py`"
- If a file was **renamed**, note both names

After writing the entry, stage and amend it into the same commit (so COMMIT-HISTORY.md is included without a second commit):
```bash
git -C <resolved-path> add <path-to-COMMIT-HISTORY.md>
git -C <resolved-path> commit --amend --no-edit
```

## Step 5 — Report Back

Tell the user:
- ✅ What was committed (short hash: `git -C <resolved-path> rev-parse --short HEAD`)
- ✅ That COMMIT-HISTORY.md was updated
- ▶️ The exact command to push: `git -C <resolved-path> push`
- If anything failed (dirty working tree, etc.), explain clearly and suggest a fix
