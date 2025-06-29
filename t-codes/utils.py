import wandb
import torch
import os

class wandb_logger:
    def __init__(self, config):
        try:
            wandb.init(
                # Set the project where this run will be logged
                project=config['project'],
                name=config['name'],
                config=config,
                entity=config['entity'],            
                )
        except:
                wandb.init(
                # Set the project where this run will be logged
                project=config.project,
                name=config.name,
                config=config,
                entity=config.entity,            
                )

        self.config = config
        self.step = None
    
    def log(self, data, step=None):
        if step is None:
            wandb.log(data)
        else:
            wandb.log(data, step=step)
            self.step = step
    
    def watch_model(self, *args, **kwargs):
        wandb.watch(*args, **kwargs)

    def log_image(self, figs):
        if self.step is None:
            wandb.log(figs)
        else:
            wandb.log(figs, step=self.step)

    def finish(self):
        wandb.finish(quiet=True)

    def load(self, net):
        path = os.path.join(self.config['path_data'], self.config['path_ckpt'], self.config['file_ckpt'])
        net.load_state_dict(torch.load(path))
        print(f'load {path}')

    def save(self, net, file_name=None):
        path_ckpt = os.path.join(self.config['path_data'], self.config['path_ckpt'])
        if not os.path.exists(path_ckpt):
            os.makedirs(path_ckpt)
            print(f'{path_ckpt} created!')

        path = os.path.join(path_ckpt, file_name)
        torch.save(net.state_dict(), path)

    def watch(self, model, log):
        wandb.watch(model, log)

    def log_temperature_metrics(self, metrics, temperature):
        """Log metrics with temperature as the x-axis"""
        # Add temperature to the metrics
        log_data = metrics.copy()
        log_data['temperature'] = temperature
        
        # Use temperature * 10 as step to ensure proper ordering in wandb
        # This makes temperature the effective x-axis in plots
        step = int(temperature * 10)
        wandb.log(log_data, step=step)
        self.step = step