echo "This script downloads pre-processed datasets used in the OpenScene project."
echo "Choose from the following options:"
echo "0 - ScanNet 3D (point clouds with GT semantic labels)"
echo "1 - ScanNet 2D (RGB-D images with camera poses)"
echo "2 - Matterport 3D (point clouds with GT semantic labels)"
echo "3 - Matterport 2D (RGB-D images with camera poses)"
echo "4 - nuScenes 3D - Validation Set (lidar point clouds with GT semantic labels)"
echo "5 - nuScenes 3D - Training Set (lidar point clouds with GT semantic labels), 379.9G"
echo "6 - nuScenes 2D (RGB images with camera poses)"
echo "7 - Replica 3D (point clouds)"
echo "8 - Replica 2D (RGB-D images)"
echo "9 - Matterport 3D with top 40 NYU classes"
echo "10 - Matterport 3D with top 80 NYU classes"
echo "11- Matterport 3D with top 160 NYU classes"
read -p "Enter dataset ID you want to download: " ds_id


if [ $ds_id == 0 ]
then
    echo "You chose 0: ScanNet 3D"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_3d.zip
    echo "Done! Start unzipping ..."
    unzip scannet_3d.zip
    echo "Done!"
elif [ $ds_id == 1 ]
then
    echo "You chose 1: ScanNet 2D"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://cvg-data.inf.ethz.ch/openscene/data/scannet_processed/scannet_2d.zip
    echo "Done! Start unzipping ..."
    unzip scannet_2d.zip
    echo "Done!"
elif [ $ds_id == 2 ]
then
    echo "You chose 2: Matterport 3D"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://cvg-data.inf.ethz.ch/openscene/data/matterport_processed/matterport_3d.zip
    echo "Done! Start unzipping ..."
    unzip matterport_3d.zip
    echo "Done!"
elif [ $ds_id == 3 ]
then
    echo "You chose 3: Matterport 2D"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://cvg-data.inf.ethz.ch/openscene/data/matterport_processed/matterport_2d.zip
    echo "Done! Start unzipping ..."
    unzip matterport_2d.zip
    echo "Done!"
elif [ $ds_id == 4 ]
then
    echo "You chose 4: nuScenes 3D - Validation Set"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://cvg-data.inf.ethz.ch/openscene/data/nuscenes_processed/nuscenes_3d.zip
    echo "Done! Start unzipping ..."
    unzip nuscenes_3d.zip
    echo "Done!"
elif [ $ds_id == 5 ]
then
    echo "You chose 5: nuScenes 3D - Training Set (379.9G)"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://cvg-data.inf.ethz.ch/openscene/data/nuscenes_processed/nuscenes_3d_train.zip
    echo "Done! Start unzipping ..."
    unzip nuscenes_3d_train.zip
    mv -r train nuscenes_3d_train
    echo "Done!"
elif [ $ds_id == 6 ]
then
    echo "You chose 6: nuScenes 2D"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://cvg-data.inf.ethz.ch/openscene/data/nuscenes_processed/nuscenes_2d.zip
    echo "Done! Start unzipping ..."
    unzip nuscenes_2d.zip
    echo "Done!"
elif [ $ds_id == 7 ]
then
    echo "You chose 7: Replica 3D"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://cvg-data.inf.ethz.ch/openscene/data/replica_processed/replica_3d.zip
    echo "Done! Start unzipping ..."
    unzip replica_3d.zip
    echo "Done!"
elif [ $ds_id == 8 ]
then
    echo "You chose 8: Replica 2D"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://cvg-data.inf.ethz.ch/openscene/data/replica_processed/replica_2d.zip
    echo "Done! Start unzipping ..."
    unzip replica_2d.zip
    echo "Done!"
elif [ $ds_id == 9 ]
then
    echo "You chose 9: Matterport 3D with top 40 NYU classes"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://cvg-data.inf.ethz.ch/openscene/data/matterport_processed/matterport_3d_40.zip
    echo "Done! Start unzipping ..."
    unzip matterport_3d_40.zip
    echo "Done!"
elif [ $ds_id == 10 ]
then
    echo "You chose 10: Matterport 3D with top 80 NYU classes"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://cvg-data.inf.ethz.ch/openscene/data/matterport_processed/matterport_3d_80.zip
    echo "Done! Start unzipping ..."
    unzip matterport_3d_80.zip
    echo "Done!"
elif [ $ds_id == 11 ]
then
    echo "You chose 11: Matterport 3D with top 160 NYU classes"
    mkdir -p data
    cd data
    echo "Start downloading ..."
    wget https://cvg-data.inf.ethz.ch/openscene/data/matterport_processed/matterport_3d_160.zip
    echo "Done! Start unzipping ..."
    unzip matterport_3d_160.zip
    echo "Done!"
else
    echo "You entered an invalid ID!"
fi
