# NH_competition

뉴스데이터를 이용한 감정 분석

## 뉴스데이터 수집


```bash
python3 run.py
```
1`finnlp`에서 제공한 `finhub-api`를 이용하여 데이터 수집 <br/>
2`STK_CSV`에서 얻어낸 `ticker data` 종목별로 데이터 수집하여 `data`파일에 `jsonl`형태로 저장<br/>
```bash
merge.ipynb
```
3.모은 2743개의 data를 하나의 json형태로 `mergedata`파일에 저장<br/>
4.저장된 json파일을 huggingface에 업로드`sehyun66/Finhub-News`의 respository에 저장<br/>


