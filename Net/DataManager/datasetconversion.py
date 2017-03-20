#
# Created on Mar 20, 2017
#
# @author: dpascualhe
#
# It converts a lmdb dataset into a h5py one.
#

import sys

import h5py
import lmdb
import numpy as np

import datum_pb2 as datum

# We initialize the cursor that we're going to use to access every element in
# the dataset.
lmdb_env = lmdb.open(sys.argv[1])
lmdb_txn = lmdb_env.begin()
lmdb_cursor = lmdb_txn.cursor()

x = []
y = []
nb_samples = 0

# Datum class deals with Google's protobuf data.
datum = datum.Datum()

if __name__ == '__main__':
    # We extract the samples and its class one by one.
    for key, value in lmdb_cursor:
        datum.ParseFromString(value)
        label = np.array(datum.label)
        data = np.array(bytearray(datum.data))
        im = data.reshape(datum.width, datum.height,
                          datum.channels).astype("uint8")
                            
        x.append(im)
        y.append(label)
        nb_samples += 1
        print("Extracted samples: " + str(nb_samples) + "\n")
    x = np.asarray(x)
    y = np.asarray(y)
    
    f = h5py.File("../Datasets/" + sys.argv[2] + ".h5", "w")
    
    # We store images
    x_dset = f.create_dataset("data", (nb_samples, datum.width, datum.height,
                                       datum.channels), dtype="f")
    x_dset[:] = x
    
    # We store labels
    y_dset = f.create_dataset("labels", (nb_samples,), dtype="i")
    y_dset[:] = y
    f.close()
    
    print("\nConversion finished.")
