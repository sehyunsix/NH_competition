
python3  -m sentiment.run \
    --dataset sehyun66/Finnhub-News \
    --subset clean \
    --split clean\
    --model mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis\
    --batchsize 1024 \
    --output_dir data/clean_data