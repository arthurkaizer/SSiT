# ==========================================================================
# Base on https://github.com/sveitser/kaggle_diabetic/blob/master/convert.py
# ==========================================================================
import os
import random
import argparse
import numpy as np

from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageFilter
from multiprocessing import Process


parser = argparse.ArgumentParser()
parser.add_argument('--image-folder', type=str, help='path to image folder')
parser.add_argument('--output-folder', type=str, help='path to output folder')
parser.add_argument('--crop-size', type=int, default=512, help='crop size of image')
parser.add_argument('-n', '--num-processes', type=int, default=8, help='number of processes to use')
parser.add_argument('--resume', action='store_true', help='resume processing from last checkpoint')
parser.add_argument('--processed-jobs-file', type=str, default='processed_jobs.txt', help='path to processed jobs file')


def main():
    args = parser.parse_args()
    image_folder = Path(args.image_folder)
    output_folder = Path(args.output_folder)

    jobs = []
    for root, _, imgs in os.walk(args.image_folder):
        root = Path(root)
        subfolders = root.relative_to(image_folder)
        output_subfolder = output_folder.joinpath(subfolders)
        output_subfolder.mkdir(parents=True, exist_ok=True)

        for img in tqdm(imgs):
            src_path = root.joinpath(img)
            tgt_path = output_subfolder.joinpath(img)
            jobs.append((src_path, tgt_path, args.crop_size))
    random.shuffle(jobs)

    # Check if resume flag is set
    if args.resume:
        processed_jobs = set()
        with open(args.processed_jobs_file, 'r') as f:
            for line in f:
                processed_jobs.add(line.strip())

        jobs = [job for job in jobs if job[0].name not in processed_jobs]

    procs = []
    job_size = len(jobs) // args.num_processes
    for i in range(args.num_processes):
        if i < args.num_processes - 1:
            procs.append(Process(target=convert_list, args=(i, jobs[i * job_size:(i + 1) * job_size], args.processed_jobs_file)))
        else:
            procs.append(Process(target=convert_list, args=(i, jobs[i * job_size:], args.processed_jobs_file)))

    for p in procs:
        p.start()

    for p in procs:
        p.join()

    print("Processamento concluído!")


def convert_list(i, jobs, processed_jobs_file):
    for j, job in enumerate(jobs):
        if j % 100 == 0:
            print(f'worker{i} has finished {j}.')

        convert(*job)
        with open(processed_jobs_file, 'a') as f:
            f.write(f"{job[0].name}\n")


def convert(fname, tgt_path, crop_size):
    img = Image.open(fname)

    blurred = img.filter(ImageFilter.BLUR)
    ba = np.array(blurred)
    h, w, _ = ba.shape

    if w > 1.2 * h:
        left_max = ba[:, : w // 32, :].max(axis=(0, 1)).astype(int)
        right_max = ba[:, - w // 32:, :].max(axis=(0, 1)).astype(int)
        max_bg = np.maximum(left_max, right_max)

        foreground = (ba > max_bg + 10).astype(np.uint8)
        bbox = Image.fromarray(foreground).getbbox()

        if bbox is None:
            print(f'bbox none for {fname} (???)')
        else:
            left, upper, right, lower = bbox
            # if we selected less than 80% of the original
            # height, just crop the square
            if right - left < 0.8 * h or lower - upper < 0.8 * h:
                print(f'bbox too small for {fname}')
                bbox = None
    else:
        bbox = None

    if bbox is None:
        bbox = square_bbox(img)

    cropped = img.crop(bbox)
    cropped = cropped.resize([crop_size, crop_size], Image.LANCZOS)
    save(cropped, tgt_path)


def square_bbox(img):
    w, h = img.size
    left = max((w - h) // 2, 0)
    upper = 0
    right = min(w - (w - h) // 2, w)
    lower = h
    return (left, upper, right, lower)


def save(img, fname):
    img.save(fname, quality=100, subsampling=0)


if __name__ == "__main__":
    main()
