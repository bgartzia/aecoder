import numpy as np


class TrainEngine(object):
    """
    Engine that launches training per epochs and episodes.
    Contains hooks to perform certain actions when necessary.
    """
    def __init__(self):
        self.hooks = {name: lambda state: None
                      for name in ['on_start',
                                   'on_start_epoch',
                                   'on_end_epoch',
                                   'on_end']}

    def train(self, loss_func, train_pipeline, val_pipeline, epochs, **kwargs):
        # State of the training procedure
        state = {
            'train_pipeline': train_pipeline,
            'val_pipeline': val_pipeline,
            'loss_func': loss_func,
            'sample': None,
            'epoch': 1,
            'total_batches': 1,
            'epochs': epochs,
            'best_val_loss': np.inf,
            'early_stopping_triggered': False
        }

        self.hooks['on_start'](state)
        for epoch in range(state['epochs']):
            self.hooks['on_start_epoch'](state)
            for batch in train_pipeline:
                state['sample'] = batch
                self.hooks['on_batch'](state)
                self.hooks['on_batch_end'](state)
                state['total_batches'] += 1

            self.hooks['on_end_epoch'](state)
            state['epoch'] += 1

            # Early stopping
            if state['early_stopping_triggered']:
                print("Early stopping triggered!")
                break

        self.hooks['on_end'](state)
        print("Training finished!")
