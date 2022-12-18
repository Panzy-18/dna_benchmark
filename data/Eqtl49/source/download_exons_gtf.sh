wget https://ftp.ensembl.org/pub/release-107/gtf/homo_sapiens/Homo_sapiens.GRCh38.107.gtf.gz
gunzip *.gz
awk '$3 == exon {print}' Homo_sapiens.GRCh38.107.gtf | sort | uniq > exons.gtf
rm -rf Homo_sapiens.GRCh38.107.gtf