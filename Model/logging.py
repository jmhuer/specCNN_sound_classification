import os
import torchaudio
from Model import classes
import webbrowser
import time

def launch(dir):
    os.system('tensorboard --logdir ' + dir + ' --port=6006')
    return

def rm_old_log(dir='logs'):
    ##launch tensorboard automatically
    os.system('lsof -i tcp:6006 | grep -v PID | awk \'{print $2}\' | xargs kill')
    os.system('rm -r ' + dir +  '/train')
    os.system('rm -r ' + dir +  '/valid')

def launchTensorBoard(dir='logs', rm_old=True):
    import threading
    if rm_old:
        rm_old_log()
    t = threading.Thread(target=launch, args=([dir]))
    t.start()
    return t


def train_log_action(logger, spectogram, pred, loss, global_step, label_id, fname):
    '''
    this function changes depending on logging task
    '''
    logger.add_scalar('loss', loss, global_step)
    if global_step % 20 == 0: ## store image after 20 global steps
        spectogram = spectogram.permute(0, 2, 1)
        waveform, sample_rate = torchaudio.load(fname)
        logger.add_audio(tag=classes[int(pred)]  + "_" + classes[label_id], snd_tensor=waveform.permute(1, 0), global_step=global_step, sample_rate=sample_rate)
        logger.add_image("ts", spectogram, global_step)
    if global_step ==5:
        webbrowser.open('http://localhost:6006/#timeseries')  # Go to example.com


def val_log_action(logger, spectogram, pred, loss, global_step, label_id, fname):
    '''
    this function changes depending on logging task
    '''
    logger.add_scalar('loss', loss, global_step)
    if global_step % 1 == 0: ## store image after 20 global steps
        spectogram = spectogram.permute(0, 2, 1)
        waveform, sample_rate = torchaudio.load(fname)
        logger.add_audio(tag=classes[int(pred)]  + "_" + classes[label_id], snd_tensor=waveform.permute(1, 0), global_step=global_step, sample_rate=sample_rate)
        logger.add_image("ts", spectogram, global_step)


def train_log_action_autoecoder(logger, image, pred,label, loss, global_step,mid_layer_sample, label_id):
    '''
    this function changes depending on logging task
    '''
    logger.add_scalar('loss', loss, global_step)
    if global_step % 20 == 0: ## store image after 20 global steps
        image = image[None]
        print("logged image")
        image = image.permute(1, 0)
        # print("code max: ", torch.max(code))
        # print("code min: ", torch.min(code))
        # logger.add_audio("_vs_" + classes[label], image, global_step=global_step, sample_rate=8000)
        # logger.add_audio(sound_name + '_recreated.wav', pred, global_step=global_step, sample_rate=8000)

        logger.add_image(classes[label_id], mid_layer_sample, global_step)



def val_log_action_autoecoder(logger, image, pred,label, loss, global_step,mid_layer_sample, label_id):
    '''
    this function changes depending on logging task
    '''
    logger.add_scalar('loss', loss, global_step)
    image = image[None]

    image = image.permute(1, 0)
    # print("code max: ", torch.max(code))
    # print("code min: ", torch.min(code))
    # print(label)
    # logger.add_audio("_vs_" + classes[label], image, global_step=global_step, sample_rate=8000)
    # logger.add_audio(sound_name + '_recreated.wav', pred, global_step=global_step, sample_rate=8000)
    logger.add_image(classes[label_id], mid_layer_sample, global_step)



