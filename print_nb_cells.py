import nbformat

nb = nbformat.read('notebook.ipynb', as_version=4)
for i, cell in enumerate(nb.cells[:10], start=1):
    print(f"=== CELL {i} ({cell.cell_type}) ===")
    print(cell.source)
    print()
