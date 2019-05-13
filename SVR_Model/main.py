from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor as MOR
import imageio as iio
import pandas as pd
import numpy as np
import joblib as jl
import time
import constants
import os
import math


def main():
    metadata = pd.read_csv("../DataGathering/testData.csv")
    images = []                 # Input array
    coord = []                  # Labels array
    start_time = time.time()
    training = 1                # Boolean: Train Models?
    csv_gen = 1                 # Boolean: Produced CSV Result Outputs for model prediction?

    # Array to shuffle data set
    np.random.seed(0)            # Get the same shuffled order each time
    rand_arr = np.random.choice((metadata.index.max() + 1), (metadata.index.max() + 1), replace=False)

    if training:
        print("Starting Model Training")

        # Construct data set for SVR
        for i in range((metadata.index.max() + 1) / 15):
            img_id = metadata.iloc[rand_arr[i], 0]
            img_1 = iio.imread("../DataGathering/Images/%d_0.jpg" % img_id)
            img_2 = iio.imread("../DataGathering/Images/%d_1.jpg" % img_id)

            images.append(np.concatenate((np.reshape(img_1, constants.IMG_LENGTH),
                                          np.reshape(img_2, constants.IMG_LENGTH))))

            coord_temp = []
            for j in range(18):
                coord_temp.append(metadata.iloc[rand_arr[i], 2 + j])

            coord.append(coord_temp)
        end_time = time.time()
        print("Image import took: {:f}s".format(end_time - start_time))

        # Train models
        num_models = 3                      # Code implemented for odd numbers(please don't enter an even one)
        parameter_increment = 10.0          # Base of exponent to multiply the base (up/down) from default

        # Default parameters: C=1, tol=1**-3, gamma=1/num_features
        c_base = 1
        tol_base = 5
        gamma_base = 1/(2.0*constants.IMG_LENGTH)

        # Train (num_models**2) many models
        for l in range(num_models):

            # Vary gamma parameter
            svr_gamma = parameter_increment ** (round(num_models / 2)-l) * gamma_base
            svr_tol = tol_base

            for k in range(num_models):

                # Vary C parameter
                svr_c = parameter_increment ** (k - round(num_models / 2)) * c_base

                # Build and train model
                print("Training model with parameter: C={}, Tol={}, Gamma={}".format(svr_c, svr_tol, svr_gamma))
                start_time = time.time()
                clf = SVR(C=svr_c, gamma=svr_gamma, tol=svr_tol, verbose=True)
                regr = MOR(clf)
                regr.fit(images, coord)
                end_time = time.time()
                print("Model generation took: {:f}s".format(end_time-start_time))
                print("\n")

                # File naming
                name_counter = 1
                filename = "SVR_Model_{}.joblib".format(name_counter)
                while os.path.isfile('./' + filename):
                    name_counter += 1
                    filename = "SVR_Model_{}.joblib".format(name_counter)

                # Saving Model
                print("Dumping to file")
                start_time = time.time()
                jl.dump(regr, filename)
                end_time = time.time()
                print("Model dump took: {:f}s".format(end_time-start_time))

                del regr

    if csv_gen:
        print("Starting Model Testing")

        # File naming
        name_counter = 1
        filename = "SVR_Model_{}".format(name_counter)
        import_name = filename + ".joblib"
        while os.path.isfile('./' + import_name):
            print("Generating CSV for " + filename)

            # Import Model
            start_time = time.time()
            regr = jl.load(import_name)
            end_time = time.time()
            print("Model import took: {:f}s".format(end_time-start_time))

            # Test Model
            num_tests = int(math.floor(((metadata.index.max() + 1) / 15)/0.8*0.2))
            tot_error = 0
            start_time = time.time()
            point_error = []
            test_values = []

            # Generate the prediction data for (num_tests) many images
            for i in range((metadata.index.max() + 1) / 15 + 1, (metadata.index.max() + 1) / 15 + num_tests+1):
                img_time = time.time()
                temp_tv = []
                temp_pe = []

                img_id = metadata.iloc[rand_arr[i], 0]
                temp_tv.append(img_id)
                temp_tv.append(metadata.iloc[rand_arr[i], 1])

                img_1 = iio.imread("../DataGathering/Images/%d_0.jpg" % img_id)
                img_2 = iio.imread("../DataGathering/Images/%d_1.jpg" % img_id)
                tester = (np.concatenate((np.reshape(img_1, constants.IMG_LENGTH),
                                          np.reshape(img_2, constants.IMG_LENGTH))))

                curr_error = 0
                val_title = ['PaX', 'PaY', 'PaZ', 'TX', 'TY', 'TZ', 'IX', 'IY', 'IZ',
                             'MX', 'MY', 'MZ', 'RX', 'RY', 'RZ', 'PiX', 'PiY', 'PiZ']
                results = regr.predict(tester.reshape(1, -1))
                for j in range(18):
                    # print("{:s}: Actual={}, Predicted={}".format(val_title[j],
                    #                                              metadata.iloc[rand_arr[i], 2+j],
                    #                                              results[0][j]))
                    temp_tv.append(metadata.iloc[rand_arr[i], 2+j])
                    temp_pe.append(results[0][j]-metadata.iloc[rand_arr[i], 2+j])
                    curr_error += (results[0][j]-metadata.iloc[rand_arr[i], 2+j])**2

                test_values.append(temp_tv)
                point_error.append(temp_pe)
                tot_error += curr_error
                img_time_end = time.time()
                print("Image MSE: {}. Image {} of {} took {:f}s".format(curr_error, i-((metadata.index.max() + 1) / 15 + 1), num_tests, img_time_end-img_time))

            end_time = time.time()
            print("\nModel Testing took: {:f}s for {} images".format(end_time - start_time, num_tests))
            print("Average MSE: {}".format(tot_error/num_tests))
            del regr

            # Save to CSV
            print("Saving to CSV")
            csv_array = np.concatenate((test_values, point_error), axis=1)
            np.savetxt(filename+".csv", csv_array, fmt="%f", delimiter=",")
            print("File Saved!")

            # Increment paths
            name_counter += 1
            filename = "SVR_Model_{}".format(name_counter)
            import_name = filename + ".joblib"


if __name__ == "__main__":
    main()
