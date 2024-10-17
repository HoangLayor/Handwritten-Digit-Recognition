from training.train import train_model
from training.evaluation import evaluate_model

if __name__ == "__main__":
    trainer = {
        'epochs': 5,
        'batch_size': 64,
        'learning_rate': 0.001,
        'model_path': './checkpoints',
        'eval': True
    }
    train_model(trainer['epochs'], trainer['batch_size'], trainer['learning_rate'], trainer['model_path'], trainer['eval'])

    evaluate_model(f'./checkpoints/mnist_model_epoch_{trainer["epochs"]}.pth', batch_size=32)