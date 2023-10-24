#!/bin/bash
#SBATCH --job-name=yt-dlp
#SBATCH --account=def-panos
#SBATCH --output="out/%x-%j.out"
#SBATCH --ntasks=4
#SBATCH --time=03:00:00
#SBATCH --mem-per-cpu=512M

# curl ifconfig.me

module load python/3.10
module list

source ENV/bin/activate

cd pages

# https://vimeo.com/user51379210
# https://vimeo.com/user51379210/videos
# yt-dlp --config-locations ~/bodycam/yt-dlp.conf https://player.vimeo.com/video/828449865?h=7ec6252795
yt-dlp --config-locations /project/6003167/slee67/bodycam/yt-dlp.conf https://vimeo.com/user51379210/videos




