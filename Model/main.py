from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor as MOR
import imageio as iio
import pandas as pd
import numpy as np
import joblib as jl
import time
import constants


def main():
    metadata = pd.read_csv("../DataGathering/testData.csv")
    images = []
    coord = []
    start_time = time.time()

    # Array to shuffle data set
    rand_arr = np.random.choice((metadata.index.max()+1), (metadata.index.max()+1), replace=False)

    # Construct data set for SVR
    for i in range((metadata.index.max()+1)/20):
        img_id = metadata.iloc[rand_arr[i], 0]
        img_1 = iio.imread("../DataGathering/Images/%d_0.jpg" % img_id)
        img_2 = iio.imread("../DataGathering/Images/%d_1.jpg" % img_id)

        images.append(np.concatenate((np.reshape(img_1, constants.IMG_LENGTH),
                                     np.reshape(img_2, constants.IMG_LENGTH))))

        coord_temp = []
        for j in range(18):
            coord_temp.append(metadata.iloc[rand_arr[i], 2+j])

        coord.append(coord_temp)
    end_time = time.time()

    print("Image import took: {:f}s".format(end_time-start_time))

    # Build and train model
    start_time = time.time()
    clf = SVR(verbose=True)
    regr = MOR(clf)
    regr.fit(images, coord)
    end_time = time.time()
    print("Model import took: {:f}s".format(end_time-start_time))
    print("\n")

    # Saving Model
    print("Dumping to file")
    start_time = time.time()
    jl.dump(regr, "SVR_Model_2.joblib")
    end_time = time.time()
    print("Model dump took: {:f}s".format(end_time-start_time))

    # Test on one image
    img_id = rand_arr[i+1]  # 135097
    img_1 = iio.imread("../DataGathering/Images/%d_0.jpg" % img_id)
    img_2 = iio.imread("../DataGathering/Images/%d_1.jpg" % img_id)
    tester = (np.concatenate((np.reshape(img_1, constants.IMG_LENGTH),
                              np.reshape(img_2, constants.IMG_LENGTH))))

    aim = []
    for j in range(18):
        aim.append([metadata.iloc[img_id, 2+j]])

    print(regr.predict(tester.reshape(1, -1)))
    print(aim)

    # debugging and testing code
    '''
    print(metadata.head(1))
    print("\n")
    
    print(metadata.iloc[0, 0])
    print(metadata.iloc[0, 1])
    print(metadata.iloc[0, 2])
    print(metadata.iloc[0, 3])
    print(metadata.iloc[0, 4])

    print(metadata.shape)
    print(metadata.axes)
    
    indx = metadata.index
    object_methods = [method_name for method_name in dir(indx)
                      if callable(getattr(indx, method_name))]
    print(object_methods)
    print(indx)
    print(indx.max())
    print((metadata.index.max())/2)
    print((metadata.index.max()+2)/2)
    '''

    # test_id = metadata.iloc[0, 0]
    # print(test_id)
    # print(type(test_id))
    # test_id = metadata.iloc[0, 0]
    #
    # img1 = iio.imread("../DataGathering/Images/%d_0.jpg" % test_id)
    # print(img1.shape[0])
    # print(img1.shape[1])


if __name__ == "__main__":
    main()
