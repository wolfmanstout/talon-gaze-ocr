---
name: commit-main
description: Commit stable-compatible changes to main, merge main into beta, and rebase any feature branch onto beta.
---

# commit-main

Commit current changes through the maintained branch stack, then update the original feature branch if applicable.

This workflow is for changes applicable to stable Talon. Commit changes that
require Talon Beta directly to `beta` instead of applying them to `main`.

## Workflow

1. **Record the current branch** and determine the situation:
   - **Uncommitted changes**: Save the full diff (`git diff`) for later verification. Stash if not on `main`, switch to `main`, pop stash, stage, and commit.
   - **Most recent commit already on current branch**: switch to `main` and cherry-pick it.
2. **Switch to `main`** and pull latest (`git pull`).
3. **Get changes onto `main`** per the situation above.
4. **Commit** (if uncommitted path) — follow the project's normal commit process. The prek pre-commit hook runs Ruff and basedpyright; retry once if auto-fixes are applied.
5. **Switch to `beta`** and pull latest (`git pull`).
6. **Merge `main` into `beta`** (`git merge main`). If conflicts arise, ask the user.
7. **If the original branch was a feature branch**:
   a. Switch back to the feature branch.
   b. Rebase onto `beta` (`git rebase beta`). If conflicts arise, ask the user.
8. **Verify no changes were lost**: Compare the affected files on the final original branch with the saved diff from step 1. If any intended hunks are missing — e.g. because they touched code that only exists on the original branch and not on `main` — re-apply those specific changes and amend the feature branch commit.
9. **End on the original branch**.

## Important notes

- Always `git pull` on `main` and `beta` before committing or merging into them.
- Never force-push. If a push is needed, confirm with the user first.
- If the prek pre-commit hook fails, check if Ruff auto-fixed issues and retry the commit once.
- Do not push any branches unless the user explicitly asks.
- If merge or rebase conflicts arise, stop and ask the user how to resolve them.
- **Feature-branch-only changes**: When uncommitted changes touch code that only exists on the feature branch (not on main), those hunks may be lost during stash/pop onto main and won't be recovered by rebasing. The verification step catches this by comparing the final state against the original diff.
