#!/bin/bash
#SBATCH --job-name=BWC_dataset_decompose
#SBATCH --nodes 1
#SBATCH --tasks-per-node=1
#SBATCH --account=def-panos
#SBATCH --cpus-per-task=1
#SBATCH --time=12:00:00
#SBATCH --output=out/log-%x-%j.out
#SBATCH --mem-per-cpu=16G

module load  StdEnv/2020
source ENV/bin/activate

OLDIFS=$IFS
IFS="
"

src="/scratch/slee67/bodycam_mp4"
dest="/scratch/slee67/bodycam_frames"
echo "Source is: ${src}"
echo "Destination is: ${dest}"
count=0
count2=0

for F in $(cat /project/6003167/slee67/bodycam/file_list.txt)	; do
	end="$dest/${F%.*}"
	echo "${end}"
	if [ ! -d "${end}" ]; then
		echo "making dir"
		mkdir "${end}"
		ffmpeg -i "${src}/${F}" "${end}/img_%05d.jpg"
		echo "finished exporting file: ${F}"
		((count2++))
	else
		echo "dir already exists"
		((count++))
	fi
done

echo "This many files already exist: $count"
echo "This many files did not exist: $count2"
IFS=$OLDIFS
echo "DONE DECOMPOSITION OF FRAMES"

