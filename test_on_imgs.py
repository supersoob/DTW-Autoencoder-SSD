from aev2a import *
from config import *
import audio_gen
import sys
import cv2
import numpy as np
import simpleaudio as saudio
import json

# TODO record key strokes for identification of soundscapes, stored with the corresponding image
# TODO so one can perform accuracy tests on soundscape identification; could also record reaction time

# shows image and corresponding sound
# select to test images from the training or test set
# select whether to choose random images from the set, or sequence from the beginning
# right arrow moves to the next image (random or next in sequence), left arrow moves to previous, esc exits
# should draw iteration by iteration and play sound at the same time
# usage: python test_on_imgs.py cfg_name test|train rand|seq model_name_postfix

if __name__ == '__main__':

    config_id = sys.argv[1] if len(sys.argv) > 1 else 'default'  # have to be defined in configs.json
    dataset = sys.argv[2] if len(sys.argv) > 2 else 'data/simple_hand.hdf5'  # path to dataset, default can be downloaded
    test_set = sys.argv[3] == 'test' if len(sys.argv) > 3 else True  # training by default
    rand_select = sys.argv[4] == 'rand' if len(sys.argv) > 4 else True
    model_name_postfix = sys.argv[5] if len(sys.argv) > 5 else ''  # if having more models with the same config

    network_params = load_config(config_id)
    network_params['batch_size'] = 1
    model_name = find_model(config_id, model_name_postfix)
    sound_len = audio_gen.soundscape_len(network_params['audio_gen'], network_params['fs'])
    model_root = 'training/'

    RIGHT_BTN = ord('d')
    LEFT_BTN = ord('a')

    print(network_params)

    # build V2A model
    model = Draw(network_params, model_name_postfix, logging=False, training=False)
    model.prepare_run_single(model_root + model_name)
    print('MODEL IS BUILT')

    # load dataset
    data = model.get_data(dataset)
    dataptr = data.root.test_img if test_set else data.root.train_img
    loadnsamples = 64

    # test model image by image
    batch_round = 0
    while True:
        # select image
        if rand_select:
            batch = model.get_batch(dataptr, batch_size=loadnsamples)
        else:
            if (batch_round+1) * loadnsamples > dataptr.shape[0]:
                batch_round = 1
            indices = np.arange(batch_round * loadnsamples, (batch_round+1) * loadnsamples)
            batch = model.get_batch(dataptr, indices=indices)

        # iterate through the batch
        img_i = 0
        while img_i < loadnsamples:
            # run model
            soundscape, gen_img, cs = model.run_single(batch[img_i], canvas_imgs_needed=True)
            soundscape = np.int16(soundscape / np.max(np.abs(soundscape)) * 32767)
            print(len(soundscape))
            cv2.imshow("Original", batch[img_i])
            if cv2.waitKey(1) == 27:
                print('EXITED'); exit(0)
            cv2.imshow("Generated", gen_img)
            if cv2.waitKey(1) == 27:
                print('EXITED'); exit(0)

            # repeat sound and drawing
            play_obj = None
            img_same = True
            while img_same:
                while play_obj and play_obj.is_playing():
                    time.sleep(0.000001)

                play_obj = saudio.play_buffer(soundscape, 2, 2, 44100)

                for i in range(model.sequence_length):
                    cv2.imshow("Decoded", cs[i])
                    c = cv2.waitKey(1)
                    if c == 27:
                        print('EXITED'); exit(0)
                    elif c == RIGHT_BTN:
                        img_i += 1; img_same = False
                        print('next image', img_i); break
                    elif c == LEFT_BTN and img_i > 0:
                        img_i -= 1; img_same = False
                        print('prev image', img_i); break
                    #time.sleep(0.00002)
                    print(sound_len)
                    time.sleep(sound_len / model.sequence_length)
                    #time.sleep((1./(sound_len/44100))/64)