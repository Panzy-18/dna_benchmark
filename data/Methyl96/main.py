from utils import generate_data

generate_data.generate(
    output_folder='./methyl96',
    source_folder='./source',
    total_groups=96,
    train_chrs=['1', '2', '3', '4', '5', '6', '7', '8', '9', '14', '15', '16', '17', '18', '19', '20', '21', '22'],
    valid_chrs=['12', '13'],
    test_chrs=['10', '11', 'X', 'Y']
)