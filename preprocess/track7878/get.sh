mkdir -p resources
cat all_files.tsv | while read line
do  
    wget https://www.encodeproject.org/files/${line}/@@download/${line}.bed.gz -O ./resources/${line}.bed.gz
done

