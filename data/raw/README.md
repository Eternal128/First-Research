# data/raw/

**Empty by design. No provider data is committed to this repository.**

Place obtained data in a subdirectory named for its source key, as registered in
`src/pcc/data/sources.py`:

```
data/raw/
├── pff_wc2022/
├── metrica_sample/
├── skillcorner_open/
└── statsbomb_open/
```

Then run:

```bash
python scripts/01_check_data_availability.py
python scripts/02_build_dataset.py --source <key> --root data/raw/<key>
```

Before relying on any source, work through its verification checklist in
`docs/data_sources.md`. Nothing about these datasets' contents, availability or
licensing has been verified by this repository, and the checklist exists so that
a write-up does not repeat provider documentation as though it were checked.

Licence terms govern use. Read them per source; several football datasets are
free to download under terms restricting commercial use, redistribution, or both.
