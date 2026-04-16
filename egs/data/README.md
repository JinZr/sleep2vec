# Example Data

Large example `.npz` files live in the Hugging Face dataset repository `zrjin/s2v_egs`, not in this Git repository.

Use the git-based sync script to clone or update that dataset repo locally and link its `.npz` files into `egs/data/npz/`:

```bash
bash egs/data/download.sh
```

The script uses:

- dataset repo: `https://huggingface.co/datasets/zrjin/s2v_egs`
- local checkout: `egs/data_hub/`
- linked files: `egs/data/npz/*.npz`

If you use git over SSH for Hugging Face, override the remote:

```bash
S2V_EGS_REMOTE=git@hf.co:datasets/zrjin/s2v_egs \
bash egs/data/download.sh
```

If `git-xet` is installed, run this once before cloning for better large-file support:

```bash
git xet install
```

`index.example.csv` is the docs example index. The real downloaded `.npz` files are local-only and stay ignored by the main repo.
