echo "This script downloads multi-view fused features used in the OpenScene project."
echo "Choose from the following options:"
echo "0 - ScanNet - Multi-view fused OpenSeg features, train/val (234.8G)"
echo "1 - ScanNet - Multi-view fused LSeg features, train/val (175.8G)"
echo "2 - Matterport - Multi-view fused OpenSeg features, train/val (198.3G)"
echo "3 - Matterport - Multi-view fused OpenSeg features, test set (66.7G)"
echo "4 - Replica - Multi-view fused OpenSeg features (9.0G)"
echo "5 - Matterport - Multi-view fused LSeg features (coming)"
echo "6 - nuScenes - Multi-view fused OpenSeg features, validation set (165G) "
echo "7 - nuScenes - Multi-view fused LSeg features (coming)"
read -p "Enter dataset ID you want to download: " ds_id


if [ $ds_id == 0 ]
then
    echo "You chose 0: ScanNet - Multi-view fused OpenSeg features"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://cvg-data.inf.ethz.ch/openscene/data/scannet_multiview_openseg.zip
    echo "Done! Start unzipping ..."
    unzip scannet_multiview_openseg.zip
    echo "Done!"
elif [ $ds_id == 1 ]
then
    echo "You chose 1: ScanNet - Multi-view fused LSeg features"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://cvg-data.inf.ethz.ch/openscene/data/scannet_multiview_lseg.zip
    echo "Done! Start unzipping ..."
    unzip scannet_multiview_lseg.zip
    echo "Done!"
elif [ $ds_id == 2 ]
then
    echo "You chose 2: Matterport - Multi-view fused OpenSeg features, train/val"
    echo "This is used only for 3D distillation"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://cvg-data.inf.ethz.ch/openscene/data/matterport_multiview_openseg.zip
    echo "Done! Start unzipping ..."
    unzip matterport_multiview_openseg.zip
    echo "Done!"
elif [ $ds_id == 3 ]
then
    echo "You chose 3: Matterport - Multi-view fused OpenSeg features, test set"
    echo "This is used for evaluation"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://cvg-data.inf.ethz.ch/openscene/data/matterport_multiview_openseg_test.zip
    echo "Done! Start unzipping ..."
    unzip matterport_multiview_openseg_test.zip
    echo "Done!"
elif [ $ds_id == 4 ]
then
    echo "You chose 4: Replica - Multi-view fused OpenSeg features"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://cvg-data.inf.ethz.ch/openscene/data/replica_multiview_openseg.zip
    echo "Done! Start unzipping ..."
    unzip replica_multiview_openseg.zip
    echo "Done!"
elif [ $ds_id == 5 ]
then
    echo "You chose 5: Matterport - Multi-view fused LSeg features"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://cvg-data.inf.ethz.ch/openscene/data/matterport_multiview_lseg.zip
    echo "Done! Start unzipping ..."
    unzip matterport_multiview_lseg.zip
    echo "Done!"
elif [ $ds_id == 6 ]
then
    echo "You chose 6: nuScenes - Multi-view fused OpenSeg features, validation set"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://cvg-data.inf.ethz.ch/openscene/data/nuscenes_multiview_openseg_val.zip
    echo "Done! Start unzipping ..."
    unzip nuscenes_multiview_openseg_val.zip
    echo "Done!"
elif [ $ds_id == 7 ]
then
    echo "You chose 7: nuScenes - Multi-view fused LSeg features"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://cvg-data.inf.ethz.ch/openscene/data/nuscenes_multiview_lseg.zip
    echo "Done! Start unzipping ..."
    unzip nuscenes_multiview_lseg.zip
    echo "Done!"
else
    echo "You entered an invalid ID!"
fi
