from __future__ import print_function
import scipy
from scipy import signal
import scipy.io.wavfile as wavfile
import scipy.fftpack
import numpy as np
from aev2a import *
import sys
import cv2
import requests
import numpy as np
import pymatlab
from skimage.transform import resize
from collections import deque
import simpleaudio as saudio
from skimage.filters import sobel
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
from audio import corr_timedomain

# TODO test
# Matlab has to run in the background if the CORF edge detection algo is used
# Run this script after starting IP Webcam app on your Android phone
#     USB tethering has to be turned on and the phone connected to PC via USB
#     Both mobile data and wifi should be turned off on the phone
# Usage: python3.6 run_proto.py test|fast corf|sobel|nothing mobile_ip cfg_id model_name_postfix


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('WRONG ARGUMENTS, CHECK SCRIPT FOR DETAILS!', file=sys.stderr)
        exit(1)

    # params
    test_run = sys.argv[1] == 'test'
    #edge_detection = sys.argv[2].lower()  # corf | sobel for now; corf is the more sophisticated algo, but requires matlab to run
    dataset = sys.argv[2]
    config_id = sys.argv[3] if len(sys.argv) > 4 else 'default'  # have to be defined in configs.json
    model_name_postfix = sys.argv[4] if len(sys.argv) > 5 else ''

    #shot_url = "http://" + mobile_ip + ":8080/shot.jpg"

    network_params = load_config(config_id)
    network_params['batch_size'] = 1
    model_name = find_model(config_id, model_name_postfix)
    sound_len = audio_gen.soundscape_len(network_params['audio_gen'], network_params['fs'])
    model_root = 'training/'

    # build V2A model
    model = Draw(network_params, model_name_postfix, logging=False, training=False)
    model.prepare_run_single(model_root + model_name)
    print('MODEL IS BUILT')

    # prepare audio
    fs = network_params['fs']
    nchannel = 2 if network_params['audio_gen']['binaural'] else 1

    batch = []
    #batch_size = 64


    # load dataset
    data = model.get_data(dataset)
    batch_size = data.root.test_img.shape[0]
    batch.append(model.get_batch(data.root.test_img, np.arange(0, batch_size)))


    for i in range(-20,21,2):
        dataset = './data/{}deg_test_hand.hdf5'.format(i)
        data = model.get_data(dataset)
        #batch_size = data.root.test_img.shape[0]
        batch.append(model.get_batch(data.root.test_img, np.arange(0, batch_size)))


    # start streaming and convertimg images
    run_times = deque([0, 0, 0], maxlen=3)
    play_obj = None
    sound_start = time.time() - 100
    try:
        play_obj = None
        img_i = 0

        while img_i < batch_size:
            if (time.time() - sound_start) < (sound_len - np.mean(run_times)):  # play_obj and play_obj.is_playing():
                time.sleep(0.00001)
                #continue

            comp_start = time.time()


            std_soundscape, gen_img2 = model.run_single(batch[0][img_i], test_run)
            std_soundscape = np.int16(std_soundscape / np.max(np.abs(std_soundscape)) * 32767)

            for x_i in range(1,22):
                # run model
                if x_i == 0:
                    continue

                if test_run:
                    soundscape, gen_img = model.run_single(batch[x_i][img_i], gen_img_needed=True)  # 0.4 rt
                else:
                    soundscape = model.run_single(batch[x_i][img_i],test_run)
                soundscape = np.int16(soundscape / np.max(np.abs(soundscape)) * 32767)
                #print(len(soundscape))

                cv2.imshow("Original", batch[x_i][img_i])
                if cv2.waitKey(1) == 27:
                    print('EXITED');
                    exit(0)
                cv2.imshow("Generated", gen_img)
                if cv2.waitKey(1) == 27:
                    print('EXITED');
                    exit(0)

                sound_start = time.time()  # time when it would start

                while play_obj and play_obj.is_playing():
                    time.sleep(0.000001)
                play_obj = saudio.play_buffer(soundscape, 2, 2, 44100)

                #pearson_corr
                #corr_timedomain.corr(std_soundscape,soundscape)

                #save audio files
                print(f'sound_img{img_i}_y{2*(x_i-11)}')
                wavfile.write('audio/' + 'sound_img{}_deg{}.wav'.format(img_i,2*(x_i-11)), 44100, soundscape)

                # measure time
                run_times.append(sound_start - comp_start)

                # logging/plotting
                print(np.mean(run_times))  # should be around 0.4 sec latency

                if cv2.waitKey(1) == ord('d'):
                    img_i += 1;
                    print('next image');
                    break;

            img_i += 1

    finally:
        pass
