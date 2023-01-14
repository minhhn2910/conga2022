# Conga2022
Code for using Qtorch+ and reproduce Conga2022 results

### Update 2023: 
For object dectection model, the torchbench and sotabench are not maintained. Follow the below steps to run the scripts:

#### Download coco dataseet and setup like : 
  1. `mkdir -p ./.data/vision/coco`
  2. `wget http://images.cocodataset.org/zips/val2017.zip`
  3. `wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip`
  4. unzip both zipped files and put them into ./.data/vision/coco
#### Install the modified version of torchbench
  1. uninstall existing torchbench if exist: `pip uninstall torchbench`
  2. `cd .. ; git clone https://github.com/minhhn2910/torchbench`
  3. `cd torchbench ; pip install -e ./`
  4. Now you can go back to conga2022 and run the script: `cd conga2022 ; python torchbench_coco.py`

#### Part of the raw data is logged in this google docs:
(this includes scale weight and scale act parameters for scaling posit, raw data logged long time a go so please be patient when reading it :D )
  https://docs.google.com/document/d/19CYdPaVoKkQGT27jPXwxmeItIZP8jLHvOOHL7lAa-Ms/edit?usp=sharing
