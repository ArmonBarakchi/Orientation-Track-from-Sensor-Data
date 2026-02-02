import pickle
import sys
import time


def load_dataset(dataset):
  cfile = f"data/trainset/cam/cam{dataset}.p"
  ifile = f"data/trainset/imu/imuRaw{dataset}.p"
  vfile = f"data/trainset/vicon/viconRot{dataset}.p"

  camd = 0
  vicd = 0
  try:
    camd = read_data(cfile)
  except:
    print("no cam data for this dataset")

  try:
    vicd = read_data(vfile)
  except:
    print("no vicd for this dataset")

  imud = read_data(ifile)

  if(camd and vicd):
    return camd, imud, vicd
  elif(vicd):
    return imud, vicd
  else:
    return camd, imud


def tic():
  return time.time()
def toc(tstart, nm=""):
  print('%s took: %s sec.\n' % (nm,(time.time() - tstart)))

def read_data(fname):
  d = []
  with open(fname, 'rb') as f:
    if sys.version_info[0] < 3:
      d = pickle.load(f)
    else:
      d = pickle.load(f, encoding='latin1')  # needed for python 3
  return d





