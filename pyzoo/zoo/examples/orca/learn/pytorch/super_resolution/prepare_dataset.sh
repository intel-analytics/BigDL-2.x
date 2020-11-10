if [ ! -f "BSDS300-images.tgz" ]; then
  wget https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/BSDS300-images.tgz
fi
if [ ! -d "dataset" ]; then
  mkdir dataset
  tar -xzf BSDS300-images.tgz -C ./dataset
fi
if [ ! -f "dataset.zip" ]; then
  zip -q -r dataset.zip dataset/
fi

