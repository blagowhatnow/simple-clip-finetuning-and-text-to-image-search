#!/usr/bin/env python

import argparse

import os
from pathlib import Path
import random
import pandas as pd



def get_data_files(args):
    caption_path = Path(args.data_dir)/args.caption_folder
    image_path = Path(args.data_dir)/args.image_folder
    _text_files = sorted(list(caption_path.glob('*.txt')))
    random.shuffle(_text_files)
    texts, images = [], []
    for txt_path in _text_files:
        img_path = (image_path/txt_path.name).with_suffix(args.image_suffix)
        if img_path.exists():
            images.append(img_path.as_posix())
            with open(txt_path) as f:
                texts.append(f.read().strip())
                
    # caption_path = Path(args.data_dir)/args.caption_folder2
    # image_path = Path(args.data_dir)/args.image_folder2
    # _text_files = sorted(list(caption_path.glob('*.txt')))
    # random.shuffle(_text_files)
    # for txt_path in _text_files:
    #     img_path = (image_path/txt_path.name).with_suffix(args.image_suffix2)
    #     if img_path.exists():
    #         images.append(img_path.as_posix())
    #         with open(txt_path) as f:
    #             texts.append(f.read().strip())

    cutt_off = int(len(texts) * args.valid_pct)
    if args.mode == "jsonl":
        import jsonlines
        with jsonlines.open(os.path.join(args.data_dir, 'train.json'), 'w') as writer:
            for image_path, caption in zip(images[cutt_off:], texts[cutt_off:]):
                writer.write({'image_path':image_path, 'caption':caption})

        with jsonlines.open(os.path.join(args.data_dir, 'val.json'), 'w') as writer:
            for image_path, caption in zip(images[:cutt_off], texts[:cutt_off]):
                writer.write({'image_path':image_path, 'caption':caption})
    
    elif args.mode == 'tsv':

        train_df = pd.DataFrame({'image_path':images[cutt_off:], 'caption':texts[cutt_off:]})
        train_df.to_csv(os.path.join(args.data_dir, 'train.tsv'), sep='\t', index=False)

        valid_df = pd.DataFrame({'image_path':images[:cutt_off], 'caption':texts[:cutt_off]})
        valid_df.to_csv(os.path.join(args.data_dir, 'val.tsv'), sep='\t', index=False)
    

if __name__=='__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', '-d')
    parser.add_argument('--image-folder', default='imagespng')
    parser.add_argument('--caption-folder', default='labelspng')
    parser.add_argument('--image-suffix', default='.png')
    # parser.add_argument('--image-suffix2', default='.png')
    # parser.add_argument('--image-folder2', default='images')
    # parser.add_argument('--caption-folder2', default='labels')
    parser.add_argument('--valid-pct', default=0.0, type=float, help='Validation persentage')
    parser.add_argument('--mode', type=str, default='tsv')
    

    args = parser.parse_args()

    get_data_files(args)
