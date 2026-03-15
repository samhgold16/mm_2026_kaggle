

# redownloading data after tournament seeds are set
```bash
cd ~/Desktop/MM_26/data/raw
kaggle competitions download -c march-machine-learning-mania-2026
unzip -o march-machine-learning-mania-2026.zip -d .
rm march-machine-learning-mania-2026.zip
```

# uploading files to kaggle competitions
```bash
kaggle competitions submit -c march-machine-learning-mania-2026 -f outputs/submission.csv -m "second attempt for baseline submission"
```