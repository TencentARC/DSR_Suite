# Data Generation
1. [Installation](#installation)
2. [Video Database Download](#video-database-download)
3. [Video Curation](#video-curation)
4. [QA Generation](#qa-generation)

## Installation
Clone the repository and install the required packages.
```bash
git clone https://github.com/TencentARC/DSR_Suite
cd data
conda create -n datagen python=3.11
conda activate datagen
pip install -r requirements.txt
```
Download the following necessary model checkpoints and put them in to the folder `models/`:
1. Grounded SAM2——sam2.1_hiera_large.pt following the instruction [HERE](https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/checkpoints/download_ckpts.sh), and groundingdino_swint_ogc.pth following the instruction [HERE](https://github.com/IDEA-Research/Grounded-SAM-2/blob/main/gdino_checkpoints/download_ckpts.sh).
2. Orient Anything——croplargeEX2/dino_weight.pt from [HERE](https://huggingface.co/Viglong/Orient-Anything/blob/main/croplargeEX2/dino_weight.pt), and dinov2-large from [HERE](https://huggingface.co/facebook/dinov2-large).
3.  π^3——checkpoint from [HERE](https://huggingface.co/yyfz233/Pi3).


## Video Database Download
The video database of our QAs is not constrained so that you can use any database as long as the videos are accompanied with captions. In the following we use [Koala-36M](https://github.com/KlingTeam/Koala-36M) as the example. Download its videos and captions. Only preserve the videos with duration between 20s to 120s. Since Koala-36M only provides the url of videos, you can download videos according to [Panda-70M](https://github.com/snap-research/Panda-70M/tree/main/dataset_dataloading).

## Video Curation
Since some videos only describe static scenes, where objects are motionless so that not suitable for dynamic spatial reasoning, we should filter out these videos first. If there are numerous videos to filter, one economical way is prompting LLMs with video captions. We use DeepSeek-R1 as the example:
```bash
python deepseek_motion.py --part_len 2 --process_num 10 --koala_csv_path ./koala_videos.csv
```
where `--part_len` specifies that intermediate results are saved after every `part_len` processed samples. `--process_num` is the number of parallel processes used for curation. `--koala_csv_path` is the path to the csv file containing captions of Koala-36M. The results will be saved in `dynamic_videos.json`.

Since orientations for non-agent objects are spurious, we further use DeepSeek-R1 to distinguish agent and non-agent objects, where the results will be saved in `agent_object.csv`:
```bash
python deepseek_agent.py --part_len 2 --process_num 10 --koala_csv_path ./koala_videos.csv
```
For more accurate filtering to obtain high-quality results and video classes, we replace DeepSeek-R1 with Gemini-2.5-Pro and prompt it with the visual contents of videos:
```bash
python gemini_motion.py --part_len 2 --process_num 10 --video_root ./video_root
```
where `--video_root` is the folder containing the videos. This should be run after obtaining `agent_object.csv` to obtain consistent results. The results will be saved in `dynamic_videos.json`.

**Note**: Calling DeepSeek-R1 and Gemini-2.5-Pro requires API. Therefore, you should replace the `'YOUR_URL'`, `'YOUR_URL'` in `deepseek_motion.py`, `deepseek_agent.py` and `gemini_motion.py` with your own token and url.
## QA Generation
Our QAs are generated with two different paradigms: template-based and non-template-based.
For generating template-based QAs, run:
```bash
python qa_temp.py --part_len 2 --process_num 10 --video_root ./video_root --qa_num 10
```
where `--part_len` specifies that intermediate results are saved after every `part_len` processed samples. `--video_root` is the folder containing the videos. `--qa_num` is the number of QAs generated for each video. The generated QAs will be saved in `qa_pairs.json`.

For generating non-template-based QAs, we prompt DeepSeek-R1 with object 3D trajectories:
```bash
python qa_nontemp.py --part_len 2 --process_num 10 --video_root ./video_root --qa_num 2
```
The results will be saved in `qa_pairs_nontemp.json`.
