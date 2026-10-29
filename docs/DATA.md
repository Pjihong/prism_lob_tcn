# FI-2010 데이터

FI-2010의 원 출처는 Ntakaris et al., *Benchmark Dataset for Mid-Price Forecasting of Limit Order Book Data with Machine Learning Methods*, Journal of Forecasting다. [저자 논문과 DOI](https://arxiv.org/abs/1705.03233)

데이터 배포 기록: [Fairdata / Etsin](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649). 배포 페이지의 현재 접근 조건을 확인한다. 고정된 다운로드 토큰을 문서에 넣지 않는다. 이번 문서 점검에서 논문 페이지는 확인했지만 배포 페이지 응답은 시간 초과되어 다운로드 자체는 확인하지 못했다.

## 저장소가 읽는 형식

```
data/FI2010/FI2010_train.csv
data/FI2010/FI2010_test.csv
```

CSV는 쉼표 구분, 헤더 있음, 첫 열은 행 인덱스다. `pandas.read_csv(..., index_col=0)` 이후 앞 40개 열을 입력으로 사용한다. 각 호가 수준의 순서는 ask price, ask volume, bid price, bid volume이며 10수준이다.

동결 파일은 인덱스 제외 149열(144개 특징과 마지막 5개 native 라벨)을 가진다. Microstructure 경로는 앞 40개 특징만 읽고 toxic/vacuum 라벨을 다시 계산한다. Native mid-price 라벨은 그 경로에 사용하지 않는다. 입력 특징의 결측 행을 임의로 삭제해 시간 순서를 이어 붙이면 안 된다.

| 파일 | 행 수 | SHA-256 |
|---|---:|---|
| FI2010_train.csv | 362400 | `53e86cb52a15a50f266a6b41532bd74e54f5cb11da288981d9aa2ecc0bc611ca` |
| FI2010_test.csv | 31937 | `fa01243fcae34e4146d6ffbed1d5f4d3f35a06afdd608c22f68f56255360d4e5` |

Development 파일 안에서 70:15를 정규화해 train/validation으로 나눈다. 별도 test를 포함한 전체 비율은 약 75.7/16.2/8.1이다. 제공된 z-score 값에 identity 입력 정규화를 적용한다.

## 정확한 재현의 현재 한계

원 배포본의 어떤 fold·기간·행 선택을 사용해 위 두 CSV를 만들었는지와 직렬화 설정을 완전히 재현하는 변환 스크립트가 현재 저장소에 없다. 임의로 공개 TXT를 전치·병합했다고 위 동결 실험과 같아지는 것은 아니다.

동결 결과를 정확히 재현하려면 위 해시의 CSV 또는 그것을 만드는 검증된 변환 이력이 필요하다. 데이터가 없는 독자는 README의 데이터 없는 테스트와 저장된 CF 재생성부터 실행할 수 있다. 변환 경로가 확보되면 선택 파일, 순서, 경계, 정규화, 행 필터링과 최종 해시를 함께 기록해야 한다.

```sh
python scripts/verify_preservation.py --include-data
```

이 검사는 파일 바이트의 일치 여부를 확인한다. 데이터 배포 권한이나 원본 변환 과정 자체를 증명하지 않는다. 데이터는 `.gitignore`로 제외되어 있다.
