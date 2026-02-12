---
name: commit-main
description: Commit changes to main, merge main into beta, and rebase feature branch onto beta if applicable.
---

# commit-main

Commit current changes to main, merge into beta, and rebase any feature branch.

## Workflow

1. **Record the current branch** and determine the situation:
   - **Uncommitted changes**: stash if on a feature branch, switch to `main`, pop stash, stage, and commit.
   - **Most recent commit already on current branch**: switch to `main` and cherry-pick it.
2. **Switch to `main`** and pull latest (`git pull`).
3. **Get changes onto `main`** per the situation above.
4. **Commit** (if uncommitted path) — follow the project's normal commit process. Pre-commit hooks run Ruff and pycheck; retry once if auto-fixes are applied.
7. **Switch to `beta`** and pull latest (`git pull`).
8. **Merge `main` into `beta`** (`git merge main`). If conflicts arise, ask the user.
9. **If the original branch was a feature branch**:
   a. Switch back to the feature branch.
   b. Rebase onto `beta` (`git rebase beta`). If conflicts arise, ask the user.
10. **End on the original branch**.

## Important notes

- Always `git pull` before committing to `main` and before merging into `beta` to avoid conflicts.
- Never force-push. If a push is needed, confirm with the user first.
- If pre-commit hooks fail, check if Ruff auto-fixed issues and retry the commit once.
- Do not push any branches unless the user explicitly asks.
- If merge or rebase conflicts arise, stop and ask the user how to resolve them.
