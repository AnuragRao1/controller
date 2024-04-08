import numpy as np  # Module that simplifies computations on matrices
import matplotlib.pyplot as plt  # Module used for plotting
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import utils  # Our own utility functions
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
INDEX_CHANNEL = [1]
# Initialize a list to store delta powers for baseline calculation
delta_powers_for_baseline = []
num_chunks_for_baseline = 10  # Number of chunks to use for baseline calculation
detected_blinks = False
iteration_number = 0
blink_detected_recently = False
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
    eeg_buffer = np.zeros((int(fs * BUFFER_LENGTH), 1))
    filter_state = None  # for use with the notch filter
    # Compute the number of epochs in "buffer_length"
    n_win_test = int(np.floor((BUFFER_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))
    #Compute the number of epochs in "buffer_length"
    cal_epochs = int(np.floor((CAL_LENGTH - EPOCH_LENGTH) /
                              SHIFT_LENGTH + 1))
    # Initialize the band power buffer (for plotting)
    # bands will be ordered: [delta, theta, alpha, beta]
    band_buffer = np.zeros((n_win_test, 4))
    cal_buffer = np.zeros((cal_epochs, 4))
    """ 3. GET DATA """
    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print('Press Ctrl-C in the console to break the while loop.')
    try:
        # track ratios and differences
        ratio = []
        differences = []
        # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
        while True:
            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(
                timeout=1, max_samples=int(SHIFT_LENGTH * fs))
            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]
            # Update EEG buffer with the new data
            eeg_buffer, filter_state = utils.update_buffer(
                eeg_buffer, ch_data, notch=True,
                filter_state=filter_state)
            """ 3.2 COMPUTE BAND POWERS """
            # Get newest samples from the buffer
            data_epoch = utils.get_last_data(eeg_buffer,
                                             EPOCH_LENGTH * fs)
            # Compute band powers
            band_powers = utils.compute_band_powers(data_epoch, fs)
            band_buffer, _ = utils.update_buffer(band_buffer,
                                                 np.asarray([band_powers]))
            # Compute the average band powers for all epochs in buffer
            # This helps to smooth out noise
            smooth_band_powers = np.mean(band_buffer, axis=0)
            print('Delta: ', band_powers[Band.Delta])
# Inside the while loop, after computing band powers
            if len(delta_powers_for_baseline) < num_chunks_for_baseline:
                # Collect delta powers until you have enough for a stable baseline
                delta_powers_for_baseline.append(band_powers[Band.Delta])
                if len(delta_powers_for_baseline) == num_chunks_for_baseline:
                    # Calculate the average delta power for the baseline
                    delta_baseline = sum(delta_powers_for_baseline) / num_chunks_for_baseline
            else:
                    # Check if current delta power exceeds the baseline by a certain threshold
                if band_powers[Band.Delta] > delta_baseline * 1.25 and not blink_detected_recently:
                    print("Blink detected!")
                    blink_detected_recently = True
                elif band_powers[Band.Delta] <= delta_baseline * 1.25:
                    blink_detected_recently = False
                    # Optionally, update the baseline if needed (e.g., every N iterations)
                    # This is just one way to update the baseline; you might choose a different strategy
                    #if update_condition_met:
                      #  delta_powers_for_baseline.pop(0)  # Remove the oldest delta power
                       # delta_powers_for_baseline.append(band_powers[Band.Delta])  # Add the newest delta power
                      #  delta_baseline = sum(delta_powers_for_baseline) / num_chunks_for_baseline  # Recalculate the baseline
            # print('Delta: ', band_powers[Band.Delta], ' Theta: ', band_powers[Band.Theta],
            #       ' Alpha: ', band_powers[Band.Alpha], ' Beta: ', band_powers[Band.Beta])
            ## Add to calibration period until full before doing analysis
            if not np.any(cal_buffer[0,:]):
                cal_buffer, _ = utils.update_buffer(cal_buffer, np.asarray([band_powers]))
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
            beta_metric = smooth_band_powers[Band.Beta] / \
                smooth_band_powers[Band.Theta]
            # print('Beta Concentration: ', beta_metric)
            ####THINGS TO TRY
            # gather baseline data for a bit, test ratio/difference from baseline as threshold
            beta_baseline = smooth_cal_band_powers[Band.Beta] / \
                smooth_cal_band_powers[Band.Theta]
            beta_ratio = beta_metric / beta_baseline
            beta_difference = beta_metric - beta_baseline
            # print('Beta baseline: ', beta_baseline)
            # print('Beta difference: ', beta_difference)
            # print('Beta ratio: ', beta_ratio)
            # ratios.append(beta_ratio)
            # differences.append(beta_difference)
            # plt.plot(ratios[])
            # plt.plot(differences[])
            # plt.show()
            # plt.show()
            # plt.clear()
            # gather baseline data for a bit, test approximate entropy
            # wavelet -> concentration
            #
    except KeyboardInterrupt:
        print('Closing!')
