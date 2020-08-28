#!/bin/bash

# Generating SceneNet images

function delete() {
	echo "Deleting the folder $1/$2"
	rm -r data/*/$1
	echo "Deletion complete"
}

while getopts "d" option;
do
	case $option in
	d)
		echo "Deleting the folders"
		delete
		exit 1
		;;
	*)
		echo "Invalid Option $OPTARG"
		exit 1
	esac
done

if (( $# != 3))
then
	echo "You must put at least 1 parameter"
	echo "Usage: ./generation.sh data_path proto_path trainNB "
	exit 1
fi



echo "Checking your arguments"
echo "Path to the dataset : $1"
echo "Path to the protobuf file : $2"
echo "Trainset number: $3"
echo "-------------------------------------------------------------"

echo "Generation SceneNet in progress on the trainset $3..."

for dos in {1..999..1}
	do
		python3 data_scenenet.py $1/$dos/ $2 $3/$dos
		#python3 rename.py $1/$dos/
done

echo "Generation SceneNet complete"
