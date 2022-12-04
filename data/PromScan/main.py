from utils import generate_data

generate_data.do_sliding_window(
    train_chrs=['1'],
    valid_chrs=['20'],
    test_chrs=['13'],
    tss_gtf_file='./transcripts.gtf',
    output_folder='./prom_scan'
)