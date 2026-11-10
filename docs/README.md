# TinyverseGP Website

This directory contains the static website for the TinyverseGP project.

## Files

| File | Description |
|---|---|
| `index.html` | Main project page (about, representations, installation, examples, team) |
| `contributing.html` | Contribution guide / call for contributions |
| `style.css` | Shared stylesheet |

## Deploying via GitHub Pages

1. Go to your repository on GitHub → **Settings** → **Pages**.
2. Under **Source**, choose the branch (e.g., `main`) and set the folder to
   `/website` (or `/docs` if you rename the directory).
3. Click **Save**. GitHub Pages will publish the site at
   `https://gpbench.github.io/TinyverseGP/` (or your configured domain).

Alternatively, move or copy the contents of this directory into the root of a
dedicated `gh-pages` branch.

## Local preview

Open `index.html` directly in a browser — no build step is needed:

```bash
# Linux / macOS
xdg-open website/index.html   # or: open website/index.html (macOS)

# Any platform with Python
python3 -m http.server 8080 --directory website
# then visit http://localhost:8080
```
