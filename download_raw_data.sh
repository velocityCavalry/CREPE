#!/bin/bash


RESET_OPT=0
while getopts ":s:" opt; do
    case ${opt} in
        s ) SPLIT=$OPTARG;;
    esac
done

mkdir -p data;
if [ -z $SPLIT ]; then # download labeled data only
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1mGzc1xmz7mHf90sI8zns0XZvNwHoHRBg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1mGzc1xmz7mHf90sI8zns0XZvNwHoHRBg" -O data/CREPE_data.zip && rm -rf /tmp/cookies.txt ;
  unzip data/CREPE_data.zip -d data
elif [ "$SPLIT" == "unlabeled" ]; then  # download unlabeled data only
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1CkLuOitxKwTPY3soRsICVRZrZyFnpeWj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1CkLuOitxKwTPY3soRsICVRZrZyFnpeWj" -O data/train_unlabeled.jsonl && rm -rf /tmp/cookies.txt;
elif [ "$SPLIT" == "all" ]; then  # download both the labeled and unlabeled data
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1mGzc1xmz7mHf90sI8zns0XZvNwHoHRBg' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1mGzc1xmz7mHf90sI8zns0XZvNwHoHRBg" -O data/CREPE_data.zip && rm -rf /tmp/cookies.txt ;
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1CkLuOitxKwTPY3soRsICVRZrZyFnpeWj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1CkLuOitxKwTPY3soRsICVRZrZyFnpeWj" -O data/train_unlabeled.jsonl && rm -rf /tmp/cookies.txt;
  unzip data/CREPE_data.zip -d data;
fi
