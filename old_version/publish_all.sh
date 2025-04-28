#!/bin/bash
# publish.sh

git checkout develop
git add .
git commit -m "publish all"
git push origin develop
git checkout main
git merge develop --no-edit
git push
git checkout develop

echo "Pushed all changes into main branch on GitHub repo"