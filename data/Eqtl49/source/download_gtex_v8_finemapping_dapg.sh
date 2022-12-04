wget https://storage.googleapis.com/gtex_analysis_v8/single_tissue_qtl_data/GTEx_v8_finemapping_DAPG.tar
tar -xf GTEx_v8_finemapping_DAPG.tar
mv GTEx_v8_finemapping_DAPG/GTEx_v8_finemapping_DAPG.txt.gz .
gunzip GTEx_v8_finemapping_DAPG.txt.gz
rm GTEx_v8_finemapping_DAPG.tar
rm -rf GTEx_v8_finemapping_DAPG

grep -v "tissue_id" GTEx_v8_finemapping_DAPG.txt | sort -k1,1 -k2,2 -k3,3n -k6,6gr > dapg_sorted.txt