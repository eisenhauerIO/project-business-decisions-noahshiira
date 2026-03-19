import nbformat

nb = nbformat.read('notebook.ipynb', as_version=4)
text = '\n'.join(cell.source for cell in nb.cells if cell.cell_type == 'code')
for token in ['pd.', 'sns.', 'mtick', 'Path(']:
    print(token, 'found' if token in text else 'NOT found')
