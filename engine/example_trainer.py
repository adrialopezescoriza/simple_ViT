# encoding: utf-8
import logging

import torch.cuda
from torch.utils.tensorboard import SummaryWriter
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.contrib.handlers import LRScheduler

writer = SummaryWriter()
def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
):
    log_period = cfg.SOLVER.LOG_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("template_model.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'accuracy': Accuracy(),
                                                            'ce_loss': Loss(loss_fn)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, 'mnist', n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer,
                              {'model': model,
                               'optimizer': optimizer})

    lr_scheduler = LRScheduler(scheduler)
    trainer.add_event_handler(Events.EPOCH_COMPLETED, lr_scheduler, {'lr_scheduler': lr_scheduler})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    RunningAverage(output_transform=lambda x: x).attach(trainer, 'avg_loss')

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        iter = (engine.state.iteration - 1) % len(train_loader) + 1

        if iter % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.4f}"
                        .format(engine.state.epoch, iter, len(train_loader), engine.state.metrics['avg_loss']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(engine):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        avg_accuracy = metrics['accuracy']
        avg_loss = metrics['ce_loss']
        writer.add_scalar("Parameters/learning_rate", lr_scheduler.get_param().as_integer_ratio(), engine.state.epoch)
        writer.add_scalar("Loss/train", avg_loss, engine.state.epoch)
        logger.info('Parameters - Epoch: {} Learning rate: {:.3f}'.format(lr_scheduler.get_param().as_integer_ratio()))
        logger.info("Training Results - Epoch: {} Avg accuracy: {:.3f} Avg Loss: {:.3f}"
                    .format(engine.state.epoch, avg_accuracy, avg_loss))

    if val_loader is not None:
        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            evaluator.run(val_loader)
            metrics = evaluator.state.metrics
            avg_accuracy = metrics['accuracy']
            avg_loss = metrics['ce_loss']
            writer.add_scalar("Loss/validation", avg_loss, engine.state.epoch)
            logger.info("Validation Results - Epoch: {} Avg accuracy: {:.3f} Avg Loss: {:.3f}"
                        .format(engine.state.epoch, avg_accuracy, avg_loss)
                        )

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        timer.reset()

    trainer.run(train_loader, max_epochs=epochs)
    writer.flush()
