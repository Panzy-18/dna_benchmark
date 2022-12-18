from tools import (
    get_config,
    get_metadata,
    Trainer,
)
from models import get_model
from datasets import get_dataset

def main():
    metadata = get_metadata()
    config = get_config()
    model = get_model(config)
    dataset = get_dataset(metadata, model.tokenizer)
    
    trainer = Trainer()
    trainer.fit(model, dataset)

if __name__ == '__main__':
    main()
