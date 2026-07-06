# BP3333 Ellipse Virtual C1 Annotated Copy

This folder is a non-destructive copy of `BP3333_ellipse_virtual_C1`.
It keeps the original BP3333 virtual-thickness C1 modelling workflow, then adds
two explicit entry points:

* `main_spyder.py` is for Spyder/debugging. Parameters are written directly in
  the file under `USER SETTINGS FOR SPYDER`.
* `main_cli.py` is for command-line and batch runs. It uses `argparse`.

`main.py` is now treated as shared workflow code. It exposes `run_one()` and
`run_all()` and is imported by both entry points.

## Typical Runs

Spyder:

```python
%runfile /Users/jiaxinkai/Desktop/p/BP3434/BP3333_ellipse_virtual_C1_dual_main_annotated/main_spyder.py
```

Command line, one airfoil:

```bash
python -m BP3333_ellipse_virtual_C1_dual_main_annotated.main_cli \
  --airfoil "BP3333_ellipse_virtual_C1_dual_main_annotated/Test Airfoils/uvblade.s1" \
  --optimize
```

Command line, all test airfoils:

```bash
python -m BP3333_ellipse_virtual_C1_dual_main_annotated.main_cli --all --optimize
```

## Reading Notes

See `CODE_READING_COMMENTS.md` for a detailed file-by-file and
function-by-function explanation.
