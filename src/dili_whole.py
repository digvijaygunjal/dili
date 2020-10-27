import pandas as pd

if __name__ == "__main__":
    dili = pd.read_csv('./data/raw/ncrt.csv', sep='\t', index_col=False)

    ncrt = pd.DataFrame()
    ncrt['smiles'] = dili['smiles']
    ncrt['severity_class'] = dili['green_annotation']

    abcd = ncrt[ncrt.apply(lambda r: not str(r['severity_class']) == 'nan', axis=1)]

    ncrt.to_csv('./data/raw/ncrt_dili.csv')
    abcd.to_csv('./data/raw/ncrt_dili_non_null.csv')
