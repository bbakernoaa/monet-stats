# GitHub Pages Deployment Workflow

This file contains the GitHub Actions workflow configuration to deploy the MkDocs documentation to GitHub Pages.

## Steps to Create the Workflow

1. Create a new file in your repository at `.github/workflows/deploy.yml`.
2. Copy the content from the "Workflow Configuration" section below into this new file.
3. Commit and push the new workflow file to your repository.

## Workflow Configuration

```yaml
name: Deploy Documentation to GitHub Pages

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

permissions:
  contents: read
  pages: write
  id-token: write

concurrency:
  group: "pages"
  cancel-in-progress: false

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
    - name: Checkout
      uses: actions/checkout@v4
    - name: Setup Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -e .[test,dev]
    - name: Build documentation
      run: |
        mkdocs build
    - name: Upload documentation artifact
      uses: actions/upload-pages-artifact@v3
      with:
        path: './site'

  deploy:
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    runs-on: ubuntu-latest
    needs: build
    steps:
    - name: Deploy to GitHub Pages
      id: deployment
      uses: actions/deploy-pages@v4
```

## Workflow Explanation

- **Trigger**: The workflow runs on pushes and pull requests to the `main` branch.
- **Permissions**: It requires write permissions for `pages` and `id-token` to deploy to GitHub Pages.
- **Concurrency**: It cancels any in-progress deployments for the same branch to avoid conflicts.
- **Build Job**:
  - Checks out the repository.
  - Sets up Python.
  - Installs project dependencies, including MkDocs and MkDocs plugins.
  - Builds the documentation using `mkdocs build`.
  - Uploads the built site as an artifact.
- **Deploy Job**:
  - Runs only on pushes to the `main` branch.
  - Deploys the artifact to GitHub Pages using the official `actions/deploy-pages` action.

## Next Steps

After creating the workflow file, you will need to enable GitHub Pages for your repository:

1. Go to your repository on GitHub.
2. Click on the **Settings** tab.
3. In the left sidebar, click on **Pages**.
4. Under "Build and deployment", select **GitHub Actions** as the source.
5. Click **Save**.

Your documentation will be automatically built and deployed to GitHub Pages every time you push to the `main` branch.