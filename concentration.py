"""
Estimate Relaxation from Band Powers

This example shows how to buffer, epoch, and transform EEG data from a single
electrode into values for each of the classic frequencies (e.g. alpha, beta, theta)
Furthermore, it shows how ratios of the band powers can be used to estimate
mental state for neurofeedback.

The neurofeedback protocols described here are inspired by
*Neurofeedback: A Comprehensive Review on System Design, Methodology and Clinical Applications* by Marzbani et. al

Adapted from https://github.com/NeuroTechX/bci-workshop
"""

import numpy as np  # Module that simplifies computations on matrices
import matplotlib.pyplot as plt  # Module used for plotting
from matplotlib.animation import FuncAnimation
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import utils  # Our own utility functions
import pandas as pd

# Handy little enum to make code more readable


class Band:
    Delta = 0
    Theta = 1
    Alpha = 2
    Beta = 3


""" EXPERIMENTAL PARAMETERS """

# Length of the EEG data buffer (in seconds)
# This buffer will hold last n seconds of data and be used for calculations
BUFFER_LENGTH = 3

# Length of the epochs used to compute the FFT (in seconds)
EPOCH_LENGTH = 1

# Amount of overlap between two consecutive epochs (in seconds)
OVERLAP_LENGTH = 0.8

# Amount to 'shift' the start of each next consecutive epoch
SHIFT_LENGTH = EPOCH_LENGTH - OVERLAP_LENGTH

# Length of the EEG data buffer for calibration (seconds)
CAL_LENGTH = 12

# Index of the channel(s) (electrodes) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL = [0,1,2,3]

if __name__ == "__main__":

    """ 1. CONNECT TO EEG STREAM """

    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    description = info.desc()

    # Get the sampling frequency
    # This is an important value that represents how many EEG data points are
    # collected in a second. This influences our frequency band calculation.
    # for the Muse 2016, this should always be 256
    fs = int(info.nominal_srate())

    """ 2. INITIALIZE BUFFERS """

    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 4))
    filter_state = None  # for use with the notch filter

    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))

    #Compute the number of epochs in "buffer_length"
    cal_epochs = int(np.floor((CAL_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))

    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, 4, 4)) # time x channel x band

    cal_buffer = np.zeros((cal_epochs, 4, 4))

    """ 3. GET DATA """

    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print('Press Ctrl-C in the console to break the while loop.')

    try:
        # track ratios and differences 
        ratios = []
        differences = []
        band_power_tracker = {}
        for i in range(4):
            for band in ['Delta', 'Theta', 'Alpha','Beta']:
                band_power_tracker[f"Channel_{str(i)}_{band}"] = []

        # fig, ax = plt.subplots()
        # line, = ax.plot([])
        # plt.show()


        # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
        while True:

            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=int(SHIFT_LENGTH * fs))

            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

            # Update EEG buffer with the new data, all channels
            for i in range(4):
                eeg_buffer[:,i], filter_state = utils.update_buffer(
                    eeg_buffer[:,i], ch_data[:,i], notch=True,
                    filter_state=filter_state)


            """ 3.2 COMPUTE BAND POWERS """
            # Get newest samples from the buffer
            data_epoch = utils.get_last_data(eeg_buffer,
                                             EPOCH_LENGTH * fs)

            # Compute band powers
            band_powers = np.zeros(4,4)
            for i in range(4):
                band_powers[i,:] = utils.compute_band_powers(data_epoch[:,i], fs)
                band_buffer[:,i,:], _ = utils.update_buffer(band_buffer[:,i,:],
                                                    np.asarray([band_powers[i,:]]))
            # Compute the average band powers for all epochs in buffer
            # This helps to smooth out noise
            smooth_band_powers = np.mean(band_buffer, axis=0)

            # add data to store in csv
            bands = ['Delta', 'Theta', 'Alpha', 'Beta']
            for i in range(4):
                for j in range(4):
                    band_power_tracker[f'Channel_{str(i)}_{bands[j]}'].append(smooth_band_powers[i,j])

            
            # print('Delta: ', band_powers[Band.Delta])
            # if band_powers[Band.Delta]>=1:
            #     print("Blink!")

            # print('Delta: ', band_powers[Band.Delta], ' Theta: ', band_powers[Band.Theta],
            #       ' Alpha: ', band_powers[Band.Alpha], ' Beta: ', band_powers[Band.Beta])


            ## Add to calibration period until full before doing analysis
            if not np.any(cal_buffer[0,:,:]):
                for i in range(4):
                    cal_buffer[:,i,:], _ = utils.update_buffer(cal_buffer[:,i,:], np.asarray([band_powers[i,:]]))
                    
                smooth_cal_band_powers = np.mean(cal_buffer, axis=0)
                print("Mean band power over calibration period: ", smooth_cal_band_powers)

                continue

    
            """ 3.3 COMPUTE NEUROFEEDBACK METRICS """
            # These metrics could also be used to drive brain-computer interfaces

            # Alpha Protocol:
            # Simple redout of alpha power, divided by delta waves in order to rule out noise
            """ alpha_metric = smooth_band_powers[Band.Alpha] / \
                smooth_band_powers[Band.Delta]
            print('Alpha Relaxation: ', alpha_metric) """

            # Beta Protocol:
            # Beta waves have been used as a measure of mental activity and concentration
            # This beta over theta ratio is commonly used as neurofeedback for ADHD
            beta_metric = np.divide(smooth_band_powers[:,Band.Beta], smooth_band_powers[:,Band.Theta])
            print('Beta Concentration: ', beta_metric)
            
            
            ####THINGS TO TRY
            # gather baseline data for a bit, test ratio/difference from baseline as threshold
            beta_baseline = np.divide(smooth_cal_band_powers[:,Band.Beta], smooth_cal_band_powers[:,Band.Theta])
            beta_ratio = np.divide(beta_metric, beta_baseline)
            beta_difference = beta_metric - beta_baseline

            #print('Beta difference: ', beta_difference)
            #print('Beta ratio: ', beta_ratio)


            ratios.append(beta_ratio)
            differences.append(beta_difference)

            #continuous change over time
            z_difference = np.divide(beta_metric - np.mean(np.array(differences, axis=0)), np.std(np.array(differences), axis=0))
            z_ratio = np.divide(beta_ratio - np.mean(np.array(ratios), axis=0), np.std(np.array(ratios), axis=0))
            
            # print("Scores: "+ str(concat_score))
            # print("Z scores for ratio: " + str((beta_ratio - np.mean(ratios)) / np.std(ratios)))

            # line.set_xdata(np.linspace(0,20,1))
            # line.set_ydata(ratios[-20:])
            # fig.canvas.draw()
            # fig.canvas.flush_events()
            # plt.pause(0.002)

            # ax.plot(ratios[-20:], label="beta_ratio")
            # ax.plot(differences[-20:], label='beta_difference')
            # ax.set_xlabel("iterations")
            # ax.legend()
            # fig.show()



            # gather baseline data for a bit, test approximate entropy
            ## TESTING WITH 1 CHANNEL FOR NOW
            channel_diff = np.array(differences)[:,0]
            channel_ratio = np.array(ratios)[:,0]
            diff_entropy = ApEn(channel_diff, 3, 0.2) # m = window size, r = distance threshold
            ratio_entropy = ApEn(channel_ratio, 3, 0.1)
            # wavelet -> concentration

            # NOTES:
            # beta ratio for channel 3 works the best

    except KeyboardInterrupt:
        df = pd.DataFrame(band_power_tracker)
        df.to_csv('band_powers_session.csv')
        print('Closing!')