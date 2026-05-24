#!/usr/bin/env bash
# Helper script: tạo repository GitHub và đẩy code.
# Sử dụng gh CLI (https://cli.github.com) hoặc fallback về git + remote đã tạo.

set -e
REPO_NAME=${1:-$(basename $(pwd))}
VISIBILITY=${2:-public} # public or private

if command -v gh >/dev/null 2>&1; then
  echo "Using gh CLI to create repo and push..."
  gh auth status || { echo "Please run 'gh auth login' first to authenticate."; exit 1; }
  gh repo create "$REPO_NAME" --$VISIBILITY --source=. --remote=origin --push
  echo "Pushed to https://github.com/$(gh api user --jq '.login')/$REPO_NAME"
else
  echo "gh CLI not found. Falling back to git only. Ensure remote origin is set."
  if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git add -A
    git commit -m "Prepare repo for GitHub" || echo "No changes to commit"
    echo "Now run: git remote add origin git@github.com:<user>/$REPO_NAME.git"
    echo "Then: git push -u origin main"
  else
    echo "Not a git repo. Initialize first: git init; git add -A; git commit -m 'init'"
  fi
fi
