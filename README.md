# coco_scripts
To setup the environment for the project to run, run the following commands:
```git clone https://github.com/jiasenlu/NeuralBabyTalk.git
cd NeuralBabyTalk
conda install pytorch=0.4.1 cuda90 -c pytorch
apt-get update && \
    apt-get install -y \
    ant \
    ca-certificates-java \
    nano \
    openjdk-8-jdk \
    python2.7 \
    unzip \
    wget && \
    apt-get clean
update-ca-certificates -f && export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64/
pip install Cython && pip install h5py \
    matplotlib \
    nltk \
    numpy \
    pycocotools \
    scikit-image \
    stanfordcorenlp \
    tensorflow \
    torchtext \
    tqdm && python -c "import nltk; nltk.download('punkt')"
pip install torchvision==0.2.0
pip install PyYAML
mkdir data/imagenet_weights && \
    cd data/imagenet_weights && \
    wget https://www.dropbox.com/sh/67fc8n6ddo3qp47/AAACkO4QntI0RPvYic5voWHFa/resnet101.pth
cd .. && \
    wget http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip && \
    unzip caption_datasets.zip && \
    mv dataset_coco.json coco/ && \
    mv dataset_flickr30k.json flickr30k/ && \
    rm caption_datasets.zip dataset_flickr8k.json
    
cd ../prepro && \
    wget https://nlp.stanford.edu/software/stanford-corenlp-full-2017-06-09.zip && \
    unzip stanford-corenlp-full-2017-06-09.zip && \
    rm stanford-corenlp-full-2017-06-09.zip

cd ../tools && \
pip install git+https://github.com/flauted/coco-caption.git@python23

cd ../data/coco && \
    wget https://www.dropbox.com/s/2gzo4ops5gbjx5h/coco_detection.h5.tar.gz && \
    tar -xzvf coco_detection.h5.tar.gz && \
    rm coco_detection.h5.tar.gz
    
cd ../.. && \
    mkdir -p save && \
    cd save && \
    wget --output-document normal_coco_1024_adam.zip https://www.dropbox.com/s/yg7vleocciocmwr/normal_coco_1024_adam.zip?dl=0 && \
    unzip normal_coco_1024_adam.zip && \
    rm normal_coco_1024_adam.zip
    
python prepro/prepro_dic_coco.py \
    --input_json data/coco/dataset_coco.json \
    --split normal \
    --output_dic_json data/coco/dic_coco.json \
    --output_cap_json data/coco/cap_coco.json && \
    python prepro/prepro_dic_coco.py \
    --input_json data/coco/dataset_coco.json \
    --split robust \
    --output_dic_json data/robust_coco/dic_coco.json \
    --output_cap_json data/robust_coco/cap_coco.json && \
    python prepro/prepro_dic_coco.py \
    --input_json data/coco/dataset_coco.json \
    --split noc \
    --output_dic_json data/noc_coco/dic_coco.json \
    --output_cap_json data/noc_coco/cap_coco.json
    
cd ../data/coco && \
wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip && \
unzip annotations_trainval2014.zip && \
mkdir images && \
cd images && \
wget http://images.cocodataset.org/zips/train2014.zip && \
wget http://images.cocodataset.org/zips/val2014.zip && \
unzip train2014.zip && \
unzip val2014.zip
    
cd ../../misc && \
rm AttModel.py && \
rm model.py && \
wget https://raw.githubusercontent.com/PengShanshan99/coco_scripts/master/AttModel.py && \
wget https://raw.githubusercontent.com/PengShanshan99/coco_scripts/master/model.py

cd .. && \
rm main.py && \
wget https://raw.githubusercontent.com/PengShanshan99/coco_scripts/master/main.py && \
wget https://raw.githubusercontent.com/PengShanshan99/coco_scripts/master/demo_shanshan_copy.py

cd .. && \
git clone https://github.com/jwyang/faster-rcnn.pytorch.git && \
cd faster-rcnn.pytorch && \
pip install -r requirements.txt && \
pip install imutils && \
mv cfgs cfgs_fr && \
wget https://raw.githubusercontent.com/PengShanshan99/coco_scripts/master/get_ppls.py && \
mkdir images_gen && \
mkdir data_fr && \
cd data_fr && \
mkdir pretrained_model && \
cd pretrained_model && \
wget --output-document faster_rcnn_coco.pth https://www.dropbox.com/s/y171ze1sdw1o2ph/faster_rcnn_1_6_9771.pth?dl=0 && \
cd lib && \
sh make.sh && \
cd ../..
```
After that, to get a caption from an image, use the `demo_simple(obj_det_model, model, img_path_name` function within `demo_shanshan_copy.py`. For a simple demo, run
```
cd NeuralBabyTalk
python demo_shanshan_copy.py  --path_opt cfgs/normal_coco_res101.yml --batch_size 1 --num_workers 0 --beam_size 3 --start_from save/normal_coco_1024_adam
```
To check the model performance on validation set, run

```python main.py --path_opt cfgs/normal_coco_res101.yml --batch_size 20 --cuda True --num_workers 0 --max_epoch 30 --inference_only True --beam_size 3 --start_from save/normal_coco_1024_adam```
