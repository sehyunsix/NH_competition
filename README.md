# NH_competition

뉴스데이터를 이용한 종목분석

## 뉴스데이터 수집
```bash
pip install -r FinNLP/requirments.txt
```
0.`finnlp`에서 제공한 `requirments.txt`설치
```bash
python3 run.py
```
1.`finnlp`에서 제공한 `finhub-api`를 이용하여 데이터 수집 <br/>
2.`STK_CSV`에서 얻어낸 `ticker data` 종목별로 데이터 수집하여 `data`파일에 `jsonl`형태로 저장<br/>
```bash
merge.ipynb
```
3.총 2743게의 종목의 data를 하나의 json형태로 `mergedata`파일에 저장<br/>
4.저장된 json파일을 huggingface에 업로드`sehyun66/Finhub-News`의 respository에 저장<br/>
```python
from datasets import load_dataset
dataset = load_dataset("sehyun66/Finnhub-News")
```
5.huggingface에 업로드된 데이터 불러오기<br/>

## Reference
`hugginface` 링크 https://huggingface.co/datasets/sehyun66/Finnhub-News<br/>
`finnlp` 링크 https://github.com/AI4Finance-Foundation/FinNLP<br/>
`finnhub`링크 https://finnhub.io/
