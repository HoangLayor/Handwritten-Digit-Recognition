import os
from training.train import train_model
from training.evaluation import eval

if __name__ == "__main__":
    trainer = {
        'epochs': 1,
        'batch_size': 64,
        'learning_rate': 0.001,
        'save_dir': 'checkpoints',
        'model_path': r'checkpoints\checkpoint_3\best accuracy\mnist_model_best.pth',
        'eval': True
    }
    train_model(
        trainer['epochs'], 
        trainer['batch_size'], 
        trainer['learning_rate'], 
        trainer['save_dir'], 
        trainer['eval']
    )

    eval(
        trainer['model_path'], 
        batch_size=32
    )