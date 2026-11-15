# Upload ChurnGuard to your existing GitHub repository

This archive is arranged with `app.py` and the folders at the repository root. Do not upload the enclosing folder as a nested `churnguard/` directory.

## Expected repository root

- `app.py`
- `train_model.py`
- `requirements.txt`
- `README.md`
- `DEPLOYMENT.md`
- `MODEL_CARD.md`
- `artifacts/` (trained model pipeline and metadata)
- `assets/` (project charts)
- `examples/` (sample account JSON)
- `tests/` (automated tests)
- `.github/workflows/` (GitHub Actions workflows)

## Recommended upload using Git

1. Extract this ZIP.
2. Clone your already-created GitHub repository:
   `git clone https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPOSITORY_NAME.git`
3. Copy the *contents* of the extracted folder into the cloned repository folder, including the hidden `.github` folder. On Windows File Explorer, enable **View > Show > Hidden items** to see `.github`.
4. Open a terminal in the cloned repository folder and run:

   ```bash
   git add .
   git status
   git commit -m "Add ChurnGuard application and model"
   git push
   ```

5. Confirm the folders appear on GitHub.

## Hugging Face sync

Open `.github/workflows/sync-to-huggingface.yml` and replace `YOUR_HF_USERNAME` in the `huggingface_repo_id` value with your Hugging Face username. Add a repository Actions secret named `HF_TOKEN` containing a Hugging Face token with write access to the Space. Never commit the token itself.

The source training workbook is intentionally excluded. Do not publish raw customer records or secrets.
