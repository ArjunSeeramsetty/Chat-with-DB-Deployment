@echo off
echo Committing frontend changes...

git add agent-ui/src/components/DataVisualization.js
git add agent-ui/package.json
git add agent-ui/README.md
git add .github/workflows/deploy-frontend.yml

git commit -m "Add enhanced dual-axis chart functionality and frontend deployment configuration"

git push origin main

echo Frontend changes committed and pushed successfully!
pause 